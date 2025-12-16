#!/usr/bin/env python3

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from matplotlib.colors import LogNorm

# Amino acid conversion dictionary
aa_3_to_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 
    'TYR': 'Y', 'VAL': 'V'
}

# Amino acid list - standard 20 amino acids
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')

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
    parser = argparse.ArgumentParser(description='Generate heatmap of error metrics per amino acid mutation.')
    parser.add_argument('pkl_file', help='Path to the pickle file containing frustration data')
    parser.add_argument('csv_file', help='Path to the CSV file containing inference results')
    parser.add_argument('--output', '-o', default='aa_error_heatmap.png', help='Output file name (default: aa_error_heatmap.png)')
    parser.add_argument('--data-output', default=None, help='Path to save the error matrix data as CSV')
    parser.add_argument('--delta', action='store_true', help='Use delta values relative to native energy')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
    parser.add_argument('--title', default=None, help='Custom plot title')
    parser.add_argument('--metric', default='rmse', choices=['mse', 'rmse', 'mae'], 
                        help='Error metric to use (mse, rmse, mae) (default: mse)')
    parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for the heatmap')
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
    
    # Calculate errors
    df['error'] = df['furstration_calc'] - df['frustration_pred']
    if args.metric == 'mse':
        df['error_metric'] = df['error'] ** 2
        metric_name = 'MSE'
    elif args.metric == 'rmse':
        df['error_metric'] = df['error'] ** 2  # Will take sqrt after grouping
        metric_name = 'RMSE'
    elif args.metric == 'mae':
        df['error_metric'] = df['error'].abs()
        metric_name = 'MAE'
    
    # Create a matrix to store error metrics for each WT -> Mut pair
    error_matrix = pd.DataFrame(0, index=AA_LIST, columns=AA_LIST)
    count_matrix = pd.DataFrame(0, index=AA_LIST, columns=AA_LIST)
    
    # Fill the matrices
    for i, row in df.iterrows():
        wt, mut = row['wildtype'], row['mutation']
        if wt in AA_LIST and mut in AA_LIST:
            error_matrix.loc[wt, mut] += row['error_metric']
            count_matrix.loc[wt, mut] += 1
    
    # Calculate average error
    error_matrix = error_matrix.div(count_matrix.replace(0, np.nan))
    
    # Apply square root for RMSE
    if args.metric == 'rmse':
        error_matrix = error_matrix.apply(np.sqrt)
    
    # Replace NaN with 0 for visualization
    error_matrix = error_matrix.fillna(0)
    
    # Save the error matrix data if requested
    if args.data_output:
        # Create a multi-index DataFrame for better representation
        result_df = pd.DataFrame()
        for wt in AA_LIST:
            for mut in AA_LIST:
                result_df = pd.concat([result_df, pd.DataFrame({
                    'wildtype': [wt],
                    'mutation': [mut],
                    f'{args.metric}_value': [error_matrix.loc[wt, mut]],
                    'count': [count_matrix.loc[wt, mut]]
                })])
        
        # Save to CSV
        result_df.to_csv(args.data_output, index=False)
        print(f"[INFO] Error matrix data saved to {args.data_output}")

    # Skip the plotting if requested
    if args.skip_plot:
        print(f"[INFO] Skipping plot generation (--skip-plot)")
        return

    # Create the plot
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    
    # Set up the colormap
    if args.log_scale:
        # For log scale, we need to avoid zero values
        min_non_zero = error_matrix[error_matrix > 0].min().min()
        error_matrix = error_matrix.replace(0, min_non_zero / 10)
        norm = LogNorm(vmin=min_non_zero / 10, vmax=error_matrix.max().max())
        cmap = 'Blues'
    else:
        norm = None
        cmap = 'Blues'
    
    # Create heatmap
    sns.heatmap(error_matrix.T, annot=True, fmt='.2f', linewidths=0.5,
                cmap=cmap, norm=norm, square=False, annot_kws={'size': 8}, ax=ax,
                cbar_kws={'shrink': 1.0, 'pad': 0.02})
    
    ax.yaxis.set_inverted(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='center')

    # Customize plot
    ax.set_xlabel('Wildtype', fontweight='normal')
    ax.set_ylabel('Mutation', fontweight='normal')

    cbar = ax.collections[0].colorbar
    cbar.set_label(f'{metric_name} {("(Log Scale)" if args.log_scale else "")}', 
                   fontsize=12, fontweight='normal', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Set the title
    if args.title:
        # Format title - replace underscores with spaces for display
        display_title = args.title.replace('_', ' ')
        
        # Split title if longer than 40 chars and contains spaces
        if len(display_title) > 80 and ' ' in display_title:
            try:
                # Find the last space before 80 chars
                split_idx = display_title[:80].rindex(' ')
                title = display_title[:split_idx] + '\n' + display_title[split_idx+1:]
            except ValueError:
                # If no space found, don't split
                title = display_title
            ax.set_title(title, fontweight='bold')
        else:
            ax.set_title(display_title, fontweight='bold')
    else:
        ax.set_title(f'Amino acid mutation {metric_name} heatmap', fontweight='bold')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"[INFO] Heatmap saved to {args.output}")

if __name__ == "__main__":
    main()