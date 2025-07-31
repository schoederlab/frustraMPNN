#!/usr/bin/env python3

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import argparse

# Amino acid conversion dictionary
aa_3_to_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 
    'TYR': 'Y', 'VAL': 'V'
}

def read_pickle(filename):
    """Read a pickle file and return its contents."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def parse_frustration_data(pklfile, delta=True):
    """Parse frustration data from pickle file."""
    data = read_pickle(pklfile)
    # Assume every pdb has only a single chain
    chains = list(data.keys())
    frust_dict = {}
    for chain in chains:
        frust_dict[chain] = {}
        for pos in data[chain]:
            singleres = data[chain][pos]
            wt = aa_3_to_1[singleres.residue_name]
            nat_energy = singleres.mutations[wt]
            mut_d = {}
            for mut in singleres.mutations.keys():
                if delta:
                    delta_val = singleres.mutations[mut] - nat_energy
                else:
                    delta_val = singleres.mutations[mut]
                mut_d[mut] = delta_val
            frust_dict[chain][f'{wt}{pos}'] = mut_d
    
    return frust_dict, chains

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate scatter plot comparing calculated and predicted frustration values.')
    parser.add_argument('pkl_file', help='Path to the pickle file containing frustration data')
    parser.add_argument('csv_file', help='Path to the CSV file containing inference results')
    parser.add_argument('--output', '-o', default='comparison_plot.png', help='Output file name (default: comparison_plot.png)')
    parser.add_argument('--data-output', default=None, help='Path to save the comparison data as CSV')
    parser.add_argument('--no-errors', action='store_true', help='Disable error metrics (RMSE, MSE) on the plot')
    parser.add_argument('--no-chain-hue', action='store_true', help='Disable coloring points by chain')
    parser.add_argument('--title', default=None, help='Custom plot title')
    parser.add_argument('--delta', action='store_true', help='Use delta values relative to native energy')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
    parser.add_argument('--skip-plot', action='store_true', help='Skip plot generation and only create dataframes')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv_file, index_col=0)
    
    # Parse frustration data
    pdb, _ = parse_frustration_data(args.pkl_file, delta=args.delta)
    
    # Add frustration calculation values to dataframe
    df['furstration_calc'] = None
    for i, row in df.iterrows():
        chain, pos, wt, mut = row['chain'], row['position'], row['wildtype'], row['mutation']
        try:
            # The position in the pickle file might be 1-indexed while the CSV might be 0-indexed
            # Try both possibilities
            try:
                df.loc[i, 'furstration_calc'] = pdb[chain][f'{wt}{pos+1}'][mut]
            except KeyError:
                df.loc[i, 'furstration_calc'] = pdb[chain][f'{wt}{pos}'][mut]
        except KeyError as e:
            print(f"Warning: Could not find data for {chain}:{wt}{pos} -> {mut}. Error: {e}")
    
    # Convert columns to numeric to avoid the "data type not inexact" error
    df['furstration_calc'] = pd.to_numeric(df['furstration_calc'])
    df['frustration_pred'] = pd.to_numeric(df['frustration_pred'])
    
    # Remove rows with NaN values
    df = df.dropna(subset=['furstration_calc', 'frustration_pred'])
    
    # Calculate min and max values for plot limits
    min_val_calc = math.floor(df['furstration_calc'].min())
    max_val_calc = math.ceil(df['furstration_calc'].max())
    min_val_pred = math.floor(df['frustration_pred'].min())
    max_val_pred = math.ceil(df['frustration_pred'].max())
    # Make the plot square by using the min and max values across both axes
    min_val = min(min_val_calc, min_val_pred)
    max_val = max(max_val_calc, max_val_pred)
    axis_val = max(abs(min_val), abs(max_val))
    min_val_calc = min_val_pred = -axis_val
    max_val_calc = max_val_pred = axis_val
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((df['furstration_calc'] - df['frustration_pred']) ** 2))
    mse = np.mean((df['furstration_calc'] - df['frustration_pred']) ** 2)
    mae = np.mean(np.abs(df['furstration_calc'] - df['frustration_pred']))
    r2 = np.corrcoef(df['furstration_calc'], df['frustration_pred'])[0, 1] ** 2
    
    # Save the comparison data if requested
    if args.data_output:
        # Add error column to dataframe
        df['error'] = df['furstration_calc'] - df['frustration_pred']
        df['error_squared'] = df['error'] ** 2
        df['error_abs'] = df['error'].abs()
        
        # Save enriched dataframe
        output_df = df.copy()
        
        # Add summary stats to a separate file
        stats_df = pd.DataFrame({
            'metric': ['MSE', 'RMSE', 'MAE', 'R^2'],
            'value': [mse, rmse, mae, r2]
        })
        
        # Save to CSV - main data
        output_df.to_csv(args.data_output, index=True)
        
        # Save stats to a separate file
        stats_file = args.data_output.replace('.csv', '_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        
        print(f"[INFO] Comparison data saved to {args.data_output}")
        print(f"[INFO] Summary statistics saved to {stats_file}")

    # Skip the plotting if requested
    if args.skip_plot:
        print(f"[INFO] Skipping plot generation (--skip-plot)")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # Set style
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    
    # Set up the scatter plot
    if args.no_chain_hue:
        sns.scatterplot(x='furstration_calc', y='frustration_pred', data=df, ax=ax,
                       s=19, alpha=0.6, zorder=10)
    else:
        sns.scatterplot(x='furstration_calc', y='frustration_pred', data=df, ax=ax,
                       hue='chain', style='chain', s=19, alpha=0.6, zorder=10)
        ax.legend(title='Chain', loc='upper left')
    
    # Set axis limits and ticks
    ax.set_xlim(min_val_calc, max_val_calc)
    ax.set_ylim(min_val_pred, max_val_pred)
    
    ax.set_xticks(range(min_val_calc, max_val_calc + 1, 1))
    ax.set_xticklabels(range(min_val_calc, max_val_calc + 1, 1))
    ax.set_yticks(range(min_val_pred, max_val_pred + 1, 1))
    ax.set_yticklabels(range(min_val_pred, max_val_pred + 1, 1))
    
    # Add grid and reference lines
    ax.grid(linestyle='--', alpha=0.7, linewidth=0.7, color='gray', zorder=0)
    ax.hlines(0, min_val_calc, max_val_calc, color='grey', linestyle='--', linewidth=1.1, zorder=0)
    ax.vlines(0, min_val_pred, max_val_pred, color='grey', linestyle='--', linewidth=1.1, zorder=0)
    
    # Add diagonal line
    diagonal_line = np.linspace(min_val_calc, max_val_calc, 100)
    ax.plot(diagonal_line, diagonal_line, color='grey', linestyle='--', linewidth=1.1, zorder=0)
    
    # Set labels and title
    ax.set_xlabel('Calculated Frustration')
    ax.set_ylabel('Predicted Frustration')
    if args.title:
        # Format title - replace underscores with spaces for display
        display_title = args.title.replace('_', ' ')
        
        # Split title if longer than 40 chars and contains spaces
        if len(display_title) > 80 and ' ' in display_title:
            try:
                # Find the last space before 80 chars
                split_idx = display_title[:80].rindex(' ')
                title = display_title[:split_idx] + '\n' + display_title[split_idx+1:]
                ax.set_title(title, fontweight='bold')
            except ValueError:
                # If no space found, don't split
                ax.set_title(display_title, fontweight='bold')
        else:
            ax.set_title(display_title, fontweight='bold')
    else:
        ax.set_title('Frustration calculation comparison', fontweight='bold')
    
    # Add error metrics if enabled
    if not args.no_errors:
        ax.text(0.95, 0.05, f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}', fontsize=12, ha='right', va='bottom',
                color='dimgray', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"[INFO] Plot saved to {args.output}")

if __name__ == "__main__":
    main() 