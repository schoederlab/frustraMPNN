#!/usr/bin/env python3

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyrosetta import *
from pyrosetta.rosetta.protocols.moves import DsspMover

# Amino acid conversion dictionary
aa_3_to_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def read_pickle(filename):
    """Read a pickle file and return its contents."""
    with open(filename, "rb") as f:
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
            frust_dict[chain][f"{wt}{pos}"] = mut_d

    return frust_dict, chains


def calc_secstructure(pose, chain: str = "A"):
    """
    Calculate the secondary structure composition of a chain from a pdbfile.
    Returns the secondary structure sequence and the fraction of helix, sheet and loops.
    """
    Dssp = DsspMover()
    Dssp.apply(pose)

    ss = pose.secstruct()
    ss_seq = [ss[i] for i in range(len(ss)) if pose.pdb_info().pose2pdb(i + 1).split()[1] == chain]
    ss_seq = "".join(ss_seq)
    return ss_seq


def get_chains(pose):
    """
    Get the chains from a pose.
    """
    chains = [v for _, v in pyrosetta.rosetta.core.pose.conf2pdb_chain(pose).items()]
    return chains


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate plots comparing errors and correlations across secondary structure elements."
    )
    parser.add_argument("pkl_file", help="Path to the pickle file containing frustration data")
    parser.add_argument("csv_file", help="Path to the CSV file containing inference results")
    parser.add_argument("pdb_file", help="Path to the PDB file for secondary structure analysis")
    parser.add_argument(
        "--output",
        "-o",
        default="error_correlation_plot.png",
        help="Output file name (default: error_correlation_plot.png)",
    )
    parser.add_argument(
        "--data-output", default=None, help="Path to save the error and correlation data as CSV"
    )
    parser.add_argument(
        "--delta", action="store_true", help="Use delta values relative to native energy"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output image (default: 300)")
    parser.add_argument("--title", default=None, help="Custom plot title")
    parser.add_argument(
        "--skip-plot", action="store_true", help="Skip plot generation and only create dataframes"
    )
    args = parser.parse_args()

    # Initialize PyRosetta
    init("-mute all", silent=True)

    # Load data
    df = pd.read_csv(args.csv_file, index_col=0)

    # Parse frustration data
    pdb, _ = parse_frustration_data(args.pkl_file, delta=args.delta)

    # Add frustration calculation values to dataframe
    df["furstration_calc"] = None
    for i, row in df.iterrows():
        chain, pos, wt, mut = row["chain"], row["position"], row["wildtype"], row["mutation"]
        try:
            # The position in the pickle file might be 1-indexed while the CSV might be 0-indexed
            # Try both possibilities
            try:
                df.loc[i, "furstration_calc"] = pdb[chain][f"{wt}{pos + 1}"][mut]
            except KeyError:
                df.loc[i, "furstration_calc"] = pdb[chain][f"{wt}{pos}"][mut]
        except KeyError as e:
            print(f"Warning: Could not find data for {chain}:{wt}{pos} -> {mut}. Error: {e}")

    # Convert columns to numeric to avoid the "data type not inexact" error
    df["furstration_calc"] = pd.to_numeric(df["furstration_calc"])
    df["frustration_pred"] = pd.to_numeric(df["frustration_pred"])

    # Remove rows with NaN values
    df = df.dropna(subset=["furstration_calc", "frustration_pred"])

    # Calculate secondary structure
    pose = pose_from_pdb(args.pdb_file)
    chains_secstruct = {chain: calc_secstructure(pose, chain) for chain in get_chains(pose)}

    # Add secondary structure information to dataframe
    df["pos_id"] = df["chain"] + "_" + df["position"].astype(str) + df["wildtype"]
    df["ss"] = None
    for pos_ids in df["pos_id"].unique():
        chain, pos = pos_ids.split("_")
        pos = pos[:-1]
        if chain not in chains_secstruct:
            continue
        ss = chains_secstruct[chain]
        if int(pos) > len(ss):
            continue
        ss_val = ss[int(pos) - 1]
        df.loc[df["pos_id"] == pos_ids, "ss"] = ss_val

    # Calculate error and correlation metrics per secondary structure
    error_per_secstruct = {
        "H": {"MSE": 0, "RMSE": 0},
        "E": {"MSE": 0, "RMSE": 0},
        "L": {"MSE": 0, "RMSE": 0},
    }

    correlation_per_secstruct = {
        "H": {r"R^2": 0, "Spearman": 0},
        "E": {r"R^2": 0, "Spearman": 0},
        "L": {r"R^2": 0, "Spearman": 0},
    }

    for ss in df["ss"].unique():
        if ss not in error_per_secstruct:
            continue
        df_ss = df[df["ss"] == ss]
        mse = ((df_ss["furstration_calc"] - df_ss["frustration_pred"]) ** 2).values
        error_per_secstruct[ss]["MSE"] = float(mse.mean())
        error_per_secstruct[ss]["RMSE"] = float(np.sqrt(mse.mean()))

        r2 = (
            np.corrcoef(df_ss["furstration_calc"].tolist(), df_ss["frustration_pred"].tolist())[
                0, 1
            ]
            ** 2
        )
        spearman = (
            df_ss[["furstration_calc", "frustration_pred"]].corr(method="spearman").iloc[0, 1]
        )
        correlation_per_secstruct[ss]["R^2"] = float(r2)
        correlation_per_secstruct[ss]["Spearman"] = float(spearman)

    # Create DataFrames for plotting
    df_error = pd.DataFrame(error_per_secstruct)
    df_corr = pd.DataFrame(correlation_per_secstruct)

    # Save the error and correlation data if requested
    if args.data_output:
        # Create a comprehensive DataFrame with all metrics
        result_data = []
        for ss in ["H", "E", "L"]:
            ss_name = {"H": "Helix", "E": "Sheet", "L": "Loops"}.get(ss, ss)
            if ss in error_per_secstruct:
                result_data.append(
                    {
                        "secstruct": ss_name,
                        "MSE": error_per_secstruct[ss]["MSE"],
                        "RMSE": error_per_secstruct[ss]["RMSE"],
                        "R^2": correlation_per_secstruct[ss]["R^2"],
                        "Spearman": correlation_per_secstruct[ss]["Spearman"],
                        "count": len(df[df["ss"] == ss]),
                    }
                )

        # Also add the overall metrics
        mse_all = ((df["furstration_calc"] - df["frustration_pred"]) ** 2).mean()
        rmse_all = np.sqrt(mse_all)
        r2_all = (
            np.corrcoef(df["furstration_calc"].tolist(), df["frustration_pred"].tolist())[0, 1] ** 2
        )
        spearman_all = (
            df[["furstration_calc", "frustration_pred"]].corr(method="spearman").iloc[0, 1]
        )

        result_data.append(
            {
                "secstruct": "All",
                "MSE": mse_all,
                "RMSE": rmse_all,
                "R^2": r2_all,
                "Spearman": spearman_all,
                "count": len(df),
            }
        )

        result_df = pd.DataFrame(result_data)

        # Get base filename without extension to create filenames
        base_output = os.path.splitext(args.data_output)[0]
        stats_output = f"{base_output}_stats.csv"
        data_output = f"{base_output}.csv"

        # Save stats to CSV
        result_df.to_csv(stats_output, index=False)
        print(f"[INFO] Error and correlation stats saved to {stats_output}")

        # Save the full dataframe with secstruct info
        df.to_csv(data_output, index=False)
        print(f"[INFO] Complete dataframe with secondary structure info saved to {data_output}")

    # Skip the plotting if requested
    if args.skip_plot:
        print("[INFO] Skipping plot generation (--skip-plot)")
        return

    # Melt the DataFrames for better visualization
    df_melted = df_error.reset_index().melt(
        id_vars="index", var_name="secstruct", value_name="Value"
    )
    df_melted.columns = ["error_type", "secstruct", "Value"]

    df_corr_melted = df_corr.reset_index().melt(
        id_vars="index", var_name="secstruct", value_name="Value"
    )
    df_corr_melted.columns = ["corr_type", "secstruct", "Value"]

    # Create the secondary structure plot
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Create figure with 3 panels (count, error, correlation)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Count of residues in each secondary structure
    # Get unique residues by position (to avoid counting each mutation separately)
    unique_residues = df[["chain", "position", "wildtype", "ss"]].drop_duplicates()
    df_counts = unique_residues["ss"].value_counts().reindex(["H", "E", "L"]).fillna(0)
    count_labels = {"H": "Helix", "E": "Sheet", "L": "Loops"}
    df_counts.index = [count_labels.get(x, x) for x in df_counts.index]

    # Create grey color palette for the count bars
    sns.barplot(x=df_counts.index, y=df_counts.values, ax=ax[0], color="darkgrey")

    # Add count values on top of bars - with offset based on 5% of maximum value
    y_max_count = df_counts.max()
    offset = y_max_count * 0.05  # 5% of maximum value
    for i, v in enumerate(df_counts.values):
        ax[0].text(i, v + offset, str(int(v)), ha="center", fontweight="normal")

    # Add 20% offset to y-limit to ensure all labels are visible
    ax[0].set_ylim(0, y_max_count * 1.2)

    ax[0].set_xlabel("")
    ax[0].set_ylabel("Number of Residues")
    ax[0].set_title("Residue Counts", fontweight="normal")
    ax[0].yaxis.grid(linestyle="--", alpha=0.7, linewidth=0.7, color="gray", zorder=0)

    # Panel 2: Error per secondary structure
    palette_error = sns.color_palette("Paired")[:2]
    sns.barplot(
        data=df_melted,
        x="secstruct",
        y="Value",
        hue="error_type",
        palette=palette_error,
        ax=ax[1],
        alpha=0.9,
    )

    # Create custom legend handles for error plot
    from matplotlib.patches import Rectangle

    legend_elements_error = [
        Rectangle((0, 0), 1, 1, facecolor=palette_error[0], alpha=0.9, label="MSE"),
        Rectangle((0, 0), 1, 1, facecolor=palette_error[1], alpha=0.9, label="RMSE"),
    ]

    # Set y-axis limit with 10% offset for error plot
    y_max = df_melted["Value"].max()
    ax[1].set_ylim(0, y_max * 1.1)  # Add 10% to the maximum value

    ax[1].set_xticks(
        range(len(df_melted["secstruct"].unique())), labels=["Helix", "Sheet", "Loops"]
    )
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    ax[1].legend(handles=legend_elements_error, loc="upper left", prop={"size": 10})
    ax[1].yaxis.grid(linestyle="--", alpha=0.7, linewidth=0.7, color="gray", zorder=0)
    ax[1].set_title("Prediction error", fontweight="normal")

    # Panel 3: Correlation per secondary structure
    palette_corr = sns.color_palette("Set2")[:2]
    bars = sns.barplot(
        data=df_corr_melted,
        x="secstruct",
        y="Value",
        hue="corr_type",
        ax=ax[2],
        alpha=0.9,
        palette=palette_corr,
    )

    # Set patterns for correlation bars
    n_groups = len(df_corr_melted["secstruct"].unique())
    for i in range(n_groups):
        # First bar of each group (R²) - no hatch
        bars.patches[i].set_hatch("")
        # Second bar of each group (Spearman) - diagonal hatch
        bars.patches[i + n_groups].set_hatch("//")

    # Create custom legend handles for correlation plot
    legend_elements_corr = [
        Rectangle((0, 0), 1, 1, facecolor=palette_corr[0], alpha=0.9, label=r"$R^2$"),
        Rectangle((0, 0), 1, 1, facecolor=palette_corr[1], hatch="//", alpha=0.9, label="Spearman"),
    ]

    ax[2].set_xticks(
        range(len(df_corr_melted["secstruct"].unique())), labels=["Helix", "Sheet", "Loops"]
    )
    ax[2].set_xlabel("")
    ax[2].set_ylabel("")
    ax[2].set_ylim(0, 1)
    ax[2].legend(handles=legend_elements_corr, loc="upper right", prop={"size": 10})
    ax[2].yaxis.grid(linestyle="--", alpha=0.7, linewidth=0.7, color="gray", zorder=0)
    ax[2].set_title("Correlation", fontweight="normal")

    # Set the main title
    if args.title:
        # Format title - replace underscores with spaces for display
        display_title = args.title.replace("_", " ")

        # Split title if longer than 120 chars and contains spaces
        if len(display_title) > 120 and " " in display_title:
            try:
                # Split at last space before 120 chars
                split_idx = display_title[:120].rindex(" ")
                title = display_title[:split_idx] + "\n" + display_title[split_idx + 1 :]
                fig.suptitle(title, fontweight="bold")
            except ValueError:
                # If no space found, don't split
                fig.suptitle(display_title, fontweight="bold")
        else:
            fig.suptitle(display_title, fontweight="bold")
    else:
        fig.suptitle("Secondary structure analysis", fontweight="bold")

    plt.tight_layout()
    # Save plot
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Plot saved to {args.output}")


if __name__ == "__main__":
    main()
