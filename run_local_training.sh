#!/bin/bash

PROJECT_NAME="frustraMPNN_fireprot"
CSV_FILE="../data_csv_frustration/fireprot/fireprot_full_data_default_frustration_raw.csv"
SPLIT_FILE="../data_csv_frustration/fireprot/splits_default/split_seed_35.pkl"
ADD_LIGHTATTENTION="True"
ADD_ESM="False"

# PROJECT_NAME=$1
# CSV_FILE=$2
# SPLIT_FILE=$3
# ADD_LIGHTATTENTION=$4
# ADD_ESM=$5

python training/train_thermompnn_refac.py training/config.yaml \
        logger="csv" \
        project="frustraMPNN_fireprot" \
        name="${PROJECT_NAME}" \
        datasets="fireprot" \
        data_loc.fireprot_csv="${CSV_FILE}" \
        data_loc.fireprot_splits="${SPLIT_FILE}" \
        data_loc.fireprot_pdbs="../pdb_files/fireprot/" \
        data_loc.weights_dir="./local_training_weights" \
        data_loc.log_dir="./local_training_log" \
        training.add_esm_embeddings="${ADD_ESM}" \
        training.esm_model="esm2_t33_650M_UR50D" \
        model.lightattn="${ADD_LIGHTATTENTION}" \
        training.epochs=10 \
		model.subtract_mut="False" \
        seed=0 \
		training.num_workers=15 \
		training.ddp="False"
