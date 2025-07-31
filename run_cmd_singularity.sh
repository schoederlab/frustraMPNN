#!/bin/bash

# ============================================================
# Default configuration values
# ============================================================
OUTFOLDER="results"
REDO_CALC=false
REDO_PLOT=false
VERBOSE=true
SKIP_INFERENCE=false
SKIP_SINGLERESIDUE=false
SKIP_DATA_ANALYSIS=false
SKIP_PLOTTING=false
LOGGING=true
TEST_MODE=false
ERROR_COUNT=0

# Variables to track timing information
INFERENCE_TIME_MS=0
SINGLERESIDUE_TIME_MS=0

# Capture original command for logging
ORIGINAL_COMMAND="$0 $*"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# ============================================================
# Helper functions
# ============================================================

# Get current date and time in specified format
get_datetime() {
    date "+%y-%m-%d | %H:%M:%S |"
}

# Get current time in milliseconds
get_time_ms() {
    echo $(($(date +%s%N)/1000000))
}

# Log message with level to both console and log file
log() {
    local level="$1"
    local msg="$2"
    local color=""
    local reset="\033[0m"
    
    case "$level" in
        "INFO")
            color="\033[0;32m" # Green
            ;;
        "WARNING")
            color="\033[0;33m" # Yellow
            ;;
        "ERROR")
            color="\033[0;31m" # Red
            ;;
        *)
            color=""
            ;;
    esac
    
    # Format log message
    local log_entry="$(get_datetime) [${level}] $msg"
    
    # Always log to file if LOG_FILE is defined
    if [ -n "$LOG_FILE" ]; then
        echo "$log_entry" >> "$LOG_FILE"
    fi
    
    # Display on console based on verbosity
    if [[ "$VERBOSE" == true ]] || [[ "$level" != "INFO" ]]; then
        echo -e "$(get_datetime) [${color}${level}${reset}] $msg"
    fi
}

# Log command execution only to log file, not to stdout
log_command() {
    local cmd="$1"
    local desc="$2"
    
    # Skip if logging is disabled
    if [ "$LOGGING" = false ] || [ -z "$LOG_FILE" ]; then
        return
    fi
    
    # Trim excessive whitespace (convert multiple spaces to single space)
    local trimmed_cmd=$(echo "$cmd" | tr -s ' ')
    
    # Format log message and write only to log file, not to stdout
    local log_entry="$(get_datetime) [COMMAND] $desc: $trimmed_cmd"
    echo "$log_entry" >> "$LOG_FILE"
}

# Check if file exists and handle redo flag
check_file_exists() {
    local file="$1"
    local redo_flag="$2"
    local description="$3"
    
    if [ -f "$file" ] && [ "$redo_flag" = false ]; then
        log "INFO" "$description exists: $file. Skipping."
        return 0
    elif [ -f "$file" ] && [ "$redo_flag" = true ]; then
        log "INFO" "$description exists but redo flag is set. Redoing."
        return 1
    else
        log "INFO" "$description doesn't exist. Creating."
        return 1
    fi
}

# Test if a command executes without error (for test mode)
test_command() {
    local cmd="$1"
    local desc="$2"
    
    log "INFO" "Testing $desc..."
    
    # Execute the command with help flag
    eval "$cmd --help" > /dev/null 2>&1
    local result=$?
    
    if [ $result -eq 0 ]; then
        log "INFO" "Test for $desc successful."
        return 0
    else
        log "ERROR" "Test for $desc failed. Command might not execute properly."
        log "ERROR" "Command: $cmd"
        # Show the output for diagnostic purposes
        eval "$cmd --help"
        ((ERROR_COUNT++))
        return 1
    fi
}

# Run a singularity command
run_singularity() {
    local binds="$1"
    local command="$2"
    local description="$3"
    local use_gpu=${4:-false}
    local track_time=${5:-false}
    local time_var_name=${6:-""}
    local critical=${7:-false}  # New parameter to indicate if this is a critical step
    
    log "INFO" "Running $description..."
    
    # Extract the first word of the command (the executable or script)
    local first_cmd=$(echo "$command" | awk '{print $1}')
    
    # Log full singularity command for reproducibility
    local singularity_cmd=""
    if [ "$use_gpu" = true ]; then
        singularity_cmd="singularity exec --nv $binds \"$CONTAINER_SIF\" $command"
    else
        singularity_cmd="singularity exec $binds \"$CONTAINER_SIF\" $command"
    fi
    log_command "$singularity_cmd" "$description"
    
    # If in test mode, check if the script runs with help
    if [ "$TEST_MODE" = true ]; then
        # For Python scripts
        if [[ "$command" == python* ]]; then
            local script_name=$(echo "$command" | grep -o "/app/scripts/[^ ]*\.py" | cut -d' ' -f1)
            if [ -n "$script_name" ]; then
                # Extract the script name without path
                local base_script=$(basename "$script_name")
                # Check if the corresponding file exists in scripts directory
                if [ -f "$(pwd)/scripts/$base_script" ]; then
                    log "INFO" "Script $base_script exists, attempting test execution..."
                    test_command "singularity exec $binds $CONTAINER_SIF python -Wignore $(pwd)/scripts/$base_script --help" "$description"
                else
                    log "ERROR" "Script $base_script not found in $(pwd)/scripts/"
                    ((ERROR_COUNT++))
                    
                    # If this is a critical test, exit immediately
                    if [ "$critical" = true ]; then
                        log "ERROR" "Critical component test failed. Exiting now."
                        exit 1
                    fi
                fi
            fi
        # For built-in commands (like singleresidue or inference) that use the entrypoint
        elif [[ "$first_cmd" == "singleresidue" ]]; then
            log "INFO" "Testing singleresidue command..."
            test_command "singularity run $binds $CONTAINER_SIF singleresidue --help" "$description"
            if [ $? -ne 0 ] && [ "$critical" = true ]; then
                log "ERROR" "Critical component test failed. Exiting now."
                exit 1
            fi
        elif [[ "$first_cmd" == "inference" ]]; then
            log "INFO" "Testing inference command..."
            test_command "singularity run $binds $CONTAINER_SIF inference --help" "$description"
            if [ $? -ne 0 ] && [ "$critical" = true ]; then
                log "ERROR" "Critical component test failed. Exiting now."
                exit 1
            fi
        else
            # For other commands
            test_command "singularity exec $binds $CONTAINER_SIF $first_cmd --help" "$description"
            if [ $? -ne 0 ] && [ "$critical" = true ]; then
                log "ERROR" "Critical component test failed. Exiting now."
                exit 1
            fi
        fi
        return 0
    fi
    
    start_time=$(date +%s)
    start_time_ms=$(get_time_ms)
    
    # Create a temporary file for capturing output
    local temp_output=$(mktemp)
    
    # Run the command - using 'run' for commands that rely on the entrypoint
    if [[ "$first_cmd" == "singleresidue" || "$first_cmd" == "inference" ]]; then
        if [ "$use_gpu" = true ]; then
            singularity run --nv $binds "$CONTAINER_SIF" $command > $temp_output 2>&1
        else
            singularity run $binds "$CONTAINER_SIF" $command > $temp_output 2>&1
        fi
    else
        # Use exec for other commands
        if [ "$use_gpu" = true ]; then
            singularity exec --nv $binds "$CONTAINER_SIF" $command > $temp_output 2>&1
        else
            singularity exec $binds "$CONTAINER_SIF" $command > $temp_output 2>&1
        fi
    fi
    
    exit_code=$?
    
    end_time=$(date +%s)
    end_time_ms=$(get_time_ms)
    
    duration=$((end_time - start_time))
    duration_ms=$((end_time_ms - start_time_ms))
    
    # Store timing information if requested
    if [ "$track_time" = true ] && [ -n "$time_var_name" ]; then
        eval "$time_var_name=$duration_ms"
        # Log in seconds for log file but store milliseconds for summary
        log "INFO" "$description completed in $duration seconds."
    else
        log "INFO" "$description completed in $duration seconds."
    fi
    
    if [ $exit_code -eq 0 ]; then
        # Display output for verbose mode
        if [[ "$VERBOSE" == true ]]; then
            cat $temp_output
        fi
        rm $temp_output
        return 0
    else
        log "ERROR" "$description failed with exit code $exit_code after $duration seconds."
        
        # Log the captured output to the log file
        if [ -n "$LOG_FILE" ]; then
            echo "=== ERROR OUTPUT START ===" >> "$LOG_FILE"
            cat $temp_output >> "$LOG_FILE"
            echo "=== ERROR OUTPUT END ===" >> "$LOG_FILE"
        fi
        
        # Display the error output to the console
        log "ERROR" "Error details:"
        cat $temp_output
        rm $temp_output
        
        ((ERROR_COUNT++))
        
        # If this is a critical step, exit immediately
        if [ "$critical" = true ]; then
            log "ERROR" "Critical step failed. Required data files will be missing. Exiting now."
            exit 1
        fi
        
        return 1
    fi
}

# Initialize log file
init_log_file() {
    # Skip if logging is disabled
    if [ "$LOGGING" = false ]; then
        return
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$OUTFOLDER/logs"
    
    # Create log file with timestamp
    LOG_FILE="$OUTFOLDER/logs/${BASENAME}_${TIMESTAMP}_run.log"
    
    # Add header to log file
    {
        echo "========================================================"
        echo "RUN LOG: $TIMESTAMP"
        echo "========================================================"
        echo "COMMAND: $ORIGINAL_COMMAND"
        echo "INPUT PDB: $INPUT_PDB"
        echo "WEIGHTS: $WEIGHTS_PATH"
        echo "CONTAINER: $CONTAINER_SIF"
        echo "OUTPUT FOLDER: $OUTFOLDER"
        echo "OPTIONS:"
        echo "  REDO_CALC: $REDO_CALC"
        echo "  REDO_PLOT: $REDO_PLOT"
        echo "  VERBOSE: $VERBOSE"
        echo "  SKIP_INFERENCE: $SKIP_INFERENCE"
        echo "  SKIP_SINGLERESIDUE: $SKIP_SINGLERESIDUE"
        echo "  SKIP_DATA_ANALYSIS: $SKIP_DATA_ANALYSIS"
        echo "  SKIP_PLOTTING: $SKIP_PLOTTING"
        echo "  LOGGING: $LOGGING"
        echo "  TEST_MODE: $TEST_MODE"
        echo "========================================================"
    } > "$LOG_FILE"
    
    log "INFO" "Log file created at $LOG_FILE"
}

# Show help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --help, -h             Show this help message"
    echo "  --input-pdb FILE       Input PDB file"
    echo "  --weights FILE         Frustra model weights file for inference" 
    echo "  --container-sif FILE   Singularity container image (.sif file)"
    echo "  --outfolder DIR        Output directory (default: $OUTFOLDER)"
    echo "  --redo-calc            Redo calculations even if output files exist (default: false)"
    echo "  --redo-plot            Redo plots even if output files exist (default: false)"
    echo "  --verbose              Show more detailed output (default: true)"
    echo "  --logging              Disable detailed logging to files for reproducibility (default: true)"
    echo "  --skip-inference       Skip the inference step (default: false)"
    echo "  --skip-singleresidue   Skip the singleresidue calculation step (default: false)"
    echo "  --skip-data-analysis   Skip all data analysis and plotting steps (default: false)"
    echo "  --skip-plotting        Skip plotting, but generate data and stats (default: false)"
    echo "  --test                 Test mode: check if scripts would run (default: false)"
}

# ============================================================
# Parse command line arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-pdb)
            INPUT_PDB="$2"
            shift 2
            ;;
        --outfolder)
            OUTFOLDER="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --container-sif)
            CONTAINER_SIF="$2"
            shift 2
            ;;
        --redo-calc)
            REDO_CALC=true
            shift
            ;;
        --redo-plot)
            REDO_PLOT=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --logging)
            LOGGING=true
            shift
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --skip-singleresidue)
            SKIP_SINGLERESIDUE=true
            shift
            ;;
        --skip-data-analysis)
            SKIP_DATA_ANALYSIS=true
            shift
            ;;
        --skip-plotting)
            SKIP_PLOTTING=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            log "ERROR" "Use --help or -h to see usage"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate inputs and create directories
# ============================================================

# Check required arguments
if [ -z "$INPUT_PDB" ]; then
    log "ERROR" "Input PDB file is required (--input-pdb)"
    exit 1
fi

if [ -z "$WEIGHTS_PATH" ]; then
    log "ERROR" "Weights file is required (--weights)"
    exit 1
fi

if [ -z "$CONTAINER_SIF" ]; then
    log "ERROR" "Singularity container file is required (--container-sif)"
    exit 1
fi

# Check if input PDB file exists
if [ ! -f "$INPUT_PDB" ]; then
    log "ERROR" "Input PDB file '$INPUT_PDB' does not exist"
    exit 1
fi

# Check if weights file exists
if [ ! -f "$WEIGHTS_PATH" ]; then
    log "ERROR" "Frustra model weights file '$WEIGHTS_PATH' does not exist"
    exit 1
fi

# Check if container file exists
if [ ! -f "$CONTAINER_SIF" ]; then
    log "ERROR" "Singularity container file '$CONTAINER_SIF' does not exist"
    exit 1
fi

# Create output folders
mkdir -p "$OUTFOLDER"
mkdir -p "$OUTFOLDER/pics"
mkdir -p "$OUTFOLDER/add_data"

log "INFO" "Output will be saved to $OUTFOLDER"
log "INFO" "Plots will be saved to $OUTFOLDER/pics"
log "INFO" "Data will be saved to $OUTFOLDER/add_data"

# ============================================================
# Setup Singularity
# ============================================================

# Check if Singularity is installed and running
if ! command -v singularity &> /dev/null; then
    log "ERROR" "Singularity is not installed or not in PATH"
    exit 1
fi

# Check if the container file is accessible
if ! singularity inspect "$CONTAINER_SIF" > /dev/null 2>&1; then
    log "ERROR" "Cannot inspect Singularity container file. It may be corrupted or not a valid SIF file."
    exit 1
fi

log "INFO" "Using Singularity container: $CONTAINER_SIF"

# ============================================================
# Setup paths
# ============================================================
WEIGHTS_DIR=$(dirname $(realpath $WEIGHTS_PATH))
PDB_DIR=$(dirname $(realpath $INPUT_PDB))
OUT_DIR=$(realpath $OUTFOLDER)
BASENAME=$(basename "$INPUT_PDB" .pdb)

# Initialize log file after BASENAME is defined
init_log_file

log "INFO" "Starting processing pipeline for $BASENAME"
log "INFO" "================================================"

# ============================================================
# Clean PDB file
# ============================================================
CLEANED_PDB="$OUT_DIR/${BASENAME}_cleaned.pdb"

if check_file_exists "$CLEANED_PDB" "$REDO_CALC" "Cleaned PDB file"; then
    INPUT_PDB="$CLEANED_PDB"
else
    log "INFO" "Cleaning PDB file"
    
    SINGULARITY_BINDS="-B ${PDB_DIR}:/app/pdbs -B ${OUT_DIR}:/app/results"
    SINGULARITY_CMD="singleresidue /app/pdbs/$(basename "$INPUT_PDB") --check_cb --clean --clean-only"
    
    run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "PDB cleaning"
    
    # Move the cleaned file to output directory
    mv -f "${PDB_DIR}/${BASENAME}_cleaned.pdb" "$OUT_DIR" 2>/dev/null
    INPUT_PDB="$CLEANED_PDB"
fi

# ============================================================
# Run inference
# ============================================================
INFERENCE_OUTPUT="$OUT_DIR/${BASENAME}_inference_results.csv"

if [ "$SKIP_INFERENCE" = false ]; then
    if check_file_exists "$INFERENCE_OUTPUT" "$REDO_CALC" "Inference output file"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results -B ${WEIGHTS_DIR}:/app/inference/weights"
        SINGULARITY_CMD="inference --pdb /app/results/$(basename "$INPUT_PDB") \
            --model_path /app/inference/weights/$(basename "$WEIGHTS_PATH") \
            --output /app/results/${BASENAME}_inference_results.csv"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Inference" true true "INFERENCE_TIME_MS" true
    fi
else
    log "INFO" "Skipping inference step (--skip-inference)"
fi

# ============================================================
# Run singleresidue frustration
# ============================================================
FRUSTRATION_PKL="$OUT_DIR/${BASENAME}_singleresidue_frustration.pkl"

if [ "$SKIP_SINGLERESIDUE" = false ]; then
    if check_file_exists "$FRUSTRATION_PKL" "$REDO_CALC" "Singleresidue frustration file"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results"
        SINGULARITY_CMD="singleresidue /app/results/$(basename "$INPUT_PDB") \
            -o /app/results/${BASENAME}_singleresidue_frustration.pkl"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Singleresidue frustration calculation" false true "SINGLERESIDUE_TIME_MS" true
    fi
else
    log "INFO" "Skipping singleresidue calculation step (--skip-singleresidue)"
fi

# ============================================================
# Generate plots (if not skipped)
# ============================================================
if [ "$SKIP_DATA_ANALYSIS" = false ]; then
    # Plot 1: Frustration comparison
    PLOT_OUTPUT="$OUTFOLDER/pics/${BASENAME}_frustration_comparison.png"
    DATA_OUTPUT="$OUTFOLDER/add_data/${BASENAME}_frustration_comparison_data.csv"
    
    if check_file_exists "$PLOT_OUTPUT" "$REDO_PLOT" "Comparison plot"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results -B $(pwd)/scripts:/app/scripts"
        PLOT_TITLE="Frustration_comparison_for_${BASENAME}"
        SKIP_PLOT_FLAG=""
        if [ "$SKIP_PLOTTING" = true ]; then
            SKIP_PLOT_FLAG="--skip-plot"
            log "INFO" "Skip-plotting mode: generating data without plots"
        fi
        SINGULARITY_CMD="python -Wignore /app/scripts/plot_comparison.py \
            /app/results/${BASENAME}_singleresidue_frustration.pkl \
            /app/results/${BASENAME}_inference_results.csv \
            --output /app/results/pics/${BASENAME}_frustration_comparison.png \
            --data-output /app/results/add_data/${BASENAME}_frustration_comparison_data.csv \
            --title ${PLOT_TITLE} \
            ${SKIP_PLOT_FLAG}"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Frustration comparison plot generation"
    fi
    
    # Plot 2: Amino acid error heatmap
    PLOT_OUTPUT="$OUTFOLDER/pics/${BASENAME}_aa_error_heatmap.png"
    DATA_OUTPUT="$OUTFOLDER/add_data/${BASENAME}_aa_error_heatmap_data.csv"
    
    if check_file_exists "$PLOT_OUTPUT" "$REDO_PLOT" "AA error heatmap"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results -B $(pwd)/scripts:/app/scripts"
        PLOT_TITLE="Prediction_error_per_amino_acid_mutation_for_${BASENAME}"
        SKIP_PLOT_FLAG=""
        if [ "$SKIP_PLOTTING" = true ]; then
            SKIP_PLOT_FLAG="--skip-plot"
        fi
        SINGULARITY_CMD="python -Wignore /app/scripts/plot_aa_error_heatmap.py \
            /app/results/${BASENAME}_singleresidue_frustration.pkl \
            /app/results/${BASENAME}_inference_results.csv \
            --output /app/results/pics/${BASENAME}_aa_error_heatmap.png \
            --data-output /app/results/add_data/${BASENAME}_aa_error_heatmap_data.csv \
            --title ${PLOT_TITLE} \
            --metric rmse \
            ${SKIP_PLOT_FLAG}"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Amino acid error heatmap generation"
    fi

    # Plot 3: Error and correlation analysis by secondary structure
    PLOT_OUTPUT="$OUTFOLDER/pics/${BASENAME}_secstruct_analysis.png"
    DATA_OUTPUT="$OUTFOLDER/add_data/${BASENAME}_secstruct_analysis_data.csv"
    
    if check_file_exists "$PLOT_OUTPUT" "$REDO_PLOT" "Secondary structure analysis plot"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results -B $(pwd)/scripts:/app/scripts"
        PLOT_TITLE="Secondary_structure_analysis_for_${BASENAME}"
        SKIP_PLOT_FLAG=""
        if [ "$SKIP_PLOTTING" = true ]; then
            SKIP_PLOT_FLAG="--skip-plot"
        fi
        SINGULARITY_CMD="python -Wignore /app/scripts/plot_secstruct_error.py \
            /app/results/${BASENAME}_singleresidue_frustration.pkl \
            /app/results/${BASENAME}_inference_results.csv \
            /app/results/$(basename "$INPUT_PDB") \
            --output /app/results/pics/${BASENAME}_secstruct_analysis.png \
            --data-output /app/results/add_data/${BASENAME}_secstruct_analysis_data.csv \
            --title ${PLOT_TITLE} \
            ${SKIP_PLOT_FLAG}"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Secondary structure analysis plot generation"
    fi
    
    # Plot 4: Location analysis (surface, boundary, core)
    PLOT_OUTPUT="$OUTFOLDER/pics/${BASENAME}_location_analysis.png"
    DATA_OUTPUT="$OUTFOLDER/add_data/${BASENAME}_location_analysis_data.csv"
    
    if check_file_exists "$PLOT_OUTPUT" "$REDO_PLOT" "Location analysis plot"; then
        : # Do nothing
    else
        SINGULARITY_BINDS="-B ${OUT_DIR}:/app/results -B $(pwd)/scripts:/app/scripts"
        PLOT_TITLE="Protein_location_analysis_for_${BASENAME}"
        SKIP_PLOT_FLAG=""
        if [ "$SKIP_PLOTTING" = true ]; then
            SKIP_PLOT_FLAG="--skip-plot"
        fi
        SINGULARITY_CMD="python -Wignore /app/scripts/plot_location_error.py \
            /app/results/${BASENAME}_singleresidue_frustration.pkl \
            /app/results/${BASENAME}_inference_results.csv \
            /app/results/$(basename "$INPUT_PDB") \
            --output /app/results/pics/${BASENAME}_location_analysis.png \
            --data-output /app/results/add_data/${BASENAME}_location_analysis_data.csv \
            --title ${PLOT_TITLE} \
            ${SKIP_PLOT_FLAG}"
        
        run_singularity "$SINGULARITY_BINDS" "$SINGULARITY_CMD" "Location analysis plot generation"
    fi
else
    log "INFO" "Skipping all data analysis and plotting steps (--skip-data-analysis)"
fi

log "INFO" "================================================"
if [ $ERROR_COUNT -eq 0 ]; then
    log "INFO" "All tasks completed successfully"
    log "INFO" "Results saved in $OUTFOLDER"
    
    # Only show log file info if logging is enabled
    if [ "$LOGGING" = true ]; then
        log "INFO" "Log file saved to $LOG_FILE"
        
        # Create a summary file with all parameters for reproducibility
        SUMMARY_FILE="$OUTFOLDER/${BASENAME}_run_summary.txt"
        {
            echo "RUN SUMMARY"
            echo "==========================================="
            echo "Date: $(date)"
            echo "PDB file: $INPUT_PDB"
            echo "Weights file: $WEIGHTS_PATH"
            echo "Singularity container: $CONTAINER_SIF"
            echo "Output folder: $OUTFOLDER"
            if [ -n "$LOG_FILE" ]; then
                echo "Run log: $LOG_FILE"
            fi
            echo "Timing Information (milliseconds):"
            echo "- Inference step: $INFERENCE_TIME_MS ms"
            echo "- Single residue frustration: $SINGLERESIDUE_TIME_MS ms"
            echo "==========================================="
        } > "$SUMMARY_FILE"
        
        log "INFO" "Summary file saved to $SUMMARY_FILE"
    fi
    
    exit 0
else
    log "ERROR" "$ERROR_COUNT task(s) failed during execution"
    log "ERROR" "Check the logs above for details"
    
    # Only show log file info if logging is enabled
    if [ "$LOGGING" = true ]; then
        log "INFO" "Log file saved to $LOG_FILE"
    fi
    
    exit 1
fi 