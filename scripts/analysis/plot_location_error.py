#!/usr/bin/env python3

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.select.residue_selector import (
    AndResidueSelector,
    ChainSelector,
    LayerSelector,
    OrResidueSelector,
)

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


def select_surface_residues(pdbfile, surface=0, boundary=0, core=0, chains: str = ""):
    """
    Select residues based on their location in the protein structure
    (surface, boundary, or core).

    Returns a comma-separated string of residues in the format 'A2,A5,A7,...'
    """
    pose = pose_from_pdb(pdbfile)
    surface_selector = LayerSelector()
    surface_selector.set_layers(core, boundary, surface)  # core, boundary, surface
    if chains != "":
        chains = chains.split(",")
        chain_selectors = OrResidueSelector()
        for chain in chains:
            chain_selectors.add_residue_selector(ChainSelector(chain))
        and_selector = AndResidueSelector(chain_selectors, surface_selector)
        surface_selector = and_selector
    res = []
    for i, b in enumerate(surface_selector.apply(pose), start=1):
        # List contains 1 if residue matches the selection criteria
        if not b:
            continue

        # Skip non-protein residues
        residue = pose.residue(i)
        if not residue.is_protein():
            continue
        res.append(str(pose.pdb_info().pose2pdb(i).replace(" ", "")))
    return ",".join([f"{r[-1]}{r[:-1]}" for r in res])


def get_residue_locations(pdbfile, chains=""):
    """
    Get the location (surface, boundary, core) for all residues in the protein.

    Returns a dictionary mapping residue identifiers to locations.
    """
    # Get residues by location
    surface_residues = select_surface_residues(
        pdbfile, surface=1, boundary=0, core=0, chains=chains
    )
    boundary_residues = select_surface_residues(
        pdbfile, surface=0, boundary=1, core=0, chains=chains
    )
    core_residues = select_surface_residues(pdbfile, surface=0, boundary=0, core=1, chains=chains)

    # Convert comma-separated strings to dictionaries for faster lookup
    surface_dict = dict.fromkeys(surface_residues.split(","), "surface") if surface_residues else {}
    boundary_dict = (
        dict.fromkeys(boundary_residues.split(","), "boundary") if boundary_residues else {}
    )
    core_dict = dict.fromkeys(core_residues.split(","), "core") if core_residues else {}

    # Combine all dictionaries
    locations = {**surface_dict, **boundary_dict, **core_dict}
    return locations


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate plots comparing errors and correlations across protein locations."
    )
    parser.add_argument("pkl_file", help="Path to the pickle file containing frustration data")
    parser.add_argument("csv_file", help="Path to the CSV file containing inference results")
    parser.add_argument("pdb_file", help="Path to the PDB file for secondary structure analysis")
    parser.add_argument(
        "--output",
        "-o",
        default="location_analysis_plot.png",
        help="Output file name (default: location_analysis_plot.png)",
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
        "--chains",
        default="",
        help="Comma-separated list of chains to analyze (default: all chains)",
    )
    parser.add_argument(
        "--skip-plot", action="store_true", help="Skip plot generation and only create dataframes"
    )
    args = parser.parse_args()

    # Initialize PyRosetta with options similar to the notebook
    init(
        "-corrections::beta_nov16 -ignore_unrecognized_res true -mute all -out:level 0", silent=True
    )

    # Load data
    df = pd.read_csv(args.csv_file, index_col=0)

    # Parse frustration data
    pdb, chains = parse_frustration_data(args.pkl_file, delta=args.delta)

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

    # Get residue locations using the provided function
    residue_locations = get_residue_locations(args.pdb_file, chains=args.chains)

    # Create a new column for residue identifiers in the format expected by residue_locations
    df["residue_id"] = df.apply(lambda row: f"{row['chain']}{row['position']}", axis=1)

    # Add location information to dataframe
    df["location"] = df["residue_id"].apply(lambda x: residue_locations.get(x, "other"))

    # Calculate error and correlation metrics per location
    location_categories = ["surface", "boundary", "core", "other"]

    error_per_location = {loc: {"MSE": 0, "RMSE": 0} for loc in location_categories}

    correlation_per_location = {loc: {r"R^2": 0, "Spearman": 0} for loc in location_categories}

    # Calculate metrics for each location
    for loc in location_categories:
        df_loc = df[df["location"] == loc]

        # Skip if no data for this location
        if len(df_loc) == 0:
            continue

        # Calculate errors
        mse = ((df_loc["furstration_calc"] - df_loc["frustration_pred"]) ** 2).values
        error_per_location[loc]["MSE"] = float(mse.mean())
        error_per_location[loc]["RMSE"] = float(np.sqrt(mse.mean()))

        # Calculate correlations
        r2 = (
            np.corrcoef(df_loc["furstration_calc"].tolist(), df_loc["frustration_pred"].tolist())[
                0, 1
            ]
            ** 2
        )
        spearman = (
            df_loc[["furstration_calc", "frustration_pred"]].corr(method="spearman").iloc[0, 1]
        )
        correlation_per_location[loc]["R^2"] = float(r2)
        correlation_per_location[loc]["Spearman"] = float(spearman)

    # Create DataFrames for plotting
    df_error_loc = pd.DataFrame(error_per_location)
    df_corr_loc = pd.DataFrame(correlation_per_location)

    # Save the error and correlation data if requested
    if args.data_output:
        # Create a comprehensive DataFrame with all metrics
        result_data = []

        # Add location metrics
        for loc in location_categories:
            if loc in error_per_location:
                count = len(df[df["location"] == loc])
                # Skip if no data for this location
                if count == 0:
                    continue

                result_data.append(
                    {
                        "location": loc.capitalize(),
                        "MSE": error_per_location[loc]["MSE"],
                        "RMSE": error_per_location[loc]["RMSE"],
                        "R^2": correlation_per_location[loc]["R^2"],
                        "Spearman": correlation_per_location[loc]["Spearman"],
                        "count": count,
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
                "location": "All",
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

        # Save the full dataframe with location info
        df.to_csv(data_output, index=False)
        print(f"[INFO] Complete dataframe with location info saved to {data_output}")

    # Skip the plotting if requested
    if args.skip_plot:
        print("[INFO] Skipping plot generation (--skip-plot)")
        return

    # Melt location DataFrames for plotting
    df_loc_melted = df_error_loc.reset_index().melt(
        id_vars="index", var_name="location", value_name="Value"
    )
    df_loc_melted.columns = ["error_type", "location", "Value"]

    df_corr_loc_melted = df_corr_loc.reset_index().melt(
        id_vars="index", var_name="location", value_name="Value"
    )
    df_corr_loc_melted.columns = ["corr_type", "location", "Value"]

    # Create the location analysis plot
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Create figure with 3 panels (count, error, correlation)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Count of residues in each location
    # Get unique residues by position (to avoid counting each mutation separately)
    unique_residues = df[["chain", "position", "wildtype", "location"]].drop_duplicates()
    df_counts = (
        unique_residues["location"]
        .value_counts()
        .reindex(["surface", "boundary", "core", "other"])
        .fillna(0)
    )
    df_counts.index = [loc.capitalize() for loc in df_counts.index]

    # Use grey color for the count bars
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

    # Panel 2: Error per location
    palette_error = sns.color_palette("Paired")[:2]
    sns.barplot(
        data=df_loc_melted,
        x="location",
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
    y_max = df_loc_melted["Value"].max()
    ax[1].set_ylim(0, y_max * 1.2)  # Add 10% to the maximum value

    ax[1].set_xticks(
        range(len(df_loc_melted["location"].unique())),
        labels=[loc.capitalize() for loc in df_loc_melted["location"].unique()],
    )
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    ax[1].legend(handles=legend_elements_error, loc="upper left", prop={"size": 10})
    ax[1].yaxis.grid(linestyle="--", alpha=0.7, linewidth=0.7, color="gray", zorder=0)
    ax[1].set_title("Prediction error", fontweight="normal")

    # Panel 3: Correlation per location
    palette_corr = sns.color_palette("Set2")[:2]
    bars_loc = sns.barplot(
        data=df_corr_loc_melted,
        x="location",
        y="Value",
        hue="corr_type",
        ax=ax[2],
        alpha=0.9,
        palette=palette_corr,
    )

    # Set patterns for correlation bars
    n_groups = len(df_corr_loc_melted["location"].unique())
    for i in range(n_groups):
        # First bar of each group (R²) - no hatch
        bars_loc.patches[i].set_hatch("")
        # Second bar of each group (Spearman) - diagonal hatch
        bars_loc.patches[i + n_groups].set_hatch("//")

    # Create custom legend handles for correlation plot
    legend_elements_corr = [
        Rectangle((0, 0), 1, 1, facecolor=palette_corr[0], alpha=0.9, label=r"$R^2$"),
        Rectangle((0, 0), 1, 1, facecolor=palette_corr[1], hatch="//", alpha=0.9, label="Spearman"),
    ]

    ax[2].set_xticks(
        range(len(df_corr_loc_melted["location"].unique())),
        labels=[loc.capitalize() for loc in df_corr_loc_melted["location"].unique()],
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
        fig.suptitle("Protein Location Analysis", fontweight="bold")

    plt.tight_layout()
    # Save plot
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Location analysis plot saved to {args.output}")


if __name__ == "__main__":
    main()
