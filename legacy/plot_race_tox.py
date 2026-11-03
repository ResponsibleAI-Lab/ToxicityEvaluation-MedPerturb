import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Okabe-Ito color palette for professional plots
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999"
]

def plot_race_barplots(base_directory):
    """
    Generates and saves bar plots comparing model metrics across different races.

    This function walks through subdirectories of a base directory, reads CSV files
    containing race-based metrics, processes the data, and generates a separate
    bar plot for each metric. The plots are saved as PNG files in their
    respective subfolders.

    Args:
        base_directory (str): The path to the main directory containing subfolders
                              with metric CSV files.
    """
    # Walk through subfolders
    for subfolder in tqdm(os.listdir(base_directory), desc="Processing folders"):
        print("now processing folders")
        folder_path = os.path.join(base_directory, subfolder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n📂 Processing folder: {folder_path}")
        csv_files = [f for f in os.listdir(folder_path) if f.endswith("race_metrics.csv")]
        all_data = []

        for file in tqdm(csv_files, desc=f"Loading CSVs in {subfolder}"):
            path = os.path.join(folder_path, file)
            df = pd.read_csv(path)

            # Normalize race labels for consistency
            df['Race'] = df['Race'].str.strip().str.title()

            # Extract a clean model name from the filename
            model_name = re.sub(
                r"[_-](temprature|temperature|nucleus|maxlen|gender_metrics).*",
                "",
                file.replace("_race_metrics.csv", ""),
                flags=re.IGNORECASE
            )
            df["Model"] = model_name
            all_data.append(df)

        if not all_data:
            print(f"No race metric CSVs found in {folder_path}. Skipping.")
            continue

        plot_df = pd.concat(all_data, ignore_index=True)
        races = sorted(plot_df["Race"].unique())
        models = sorted(plot_df["Model"].unique())
        metrics = plot_df["Metric"].unique()

        for metric in tqdm(metrics, desc=f"Plotting in {subfolder}"):
            metric_df = plot_df[plot_df["Metric"] == metric]

            # Prepare bar positions for grouped bar chart
            n_races = len(races)
            n_models = len(models)
            bar_width = 0.8 / n_models
            x = np.arange(n_races)

            # --- MODIFICATION: Increased figure height from 8 to 10 ---
            plt.figure(figsize=(14, 12)) 
            for i, model in enumerate(models):
                model_data = metric_df[metric_df["Model"] == model].set_index("Race").reindex(races)
                means = model_data["Mean"].values
                ci_low = model_data["CI_low"].values
                ci_high = model_data["CI_high"].values

                # Calculate error intervals, ensuring they are non-negative
                lower_err = np.clip(means - ci_low, 0, None)
                upper_err = np.clip(ci_high - means, 0, None)
                yerr = [lower_err, upper_err]

                plt.bar(
                    x + i * bar_width,
                    means,
                    width=bar_width,
                    yerr=yerr,
                    capsize=4,
                    label=model,
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    alpha=0.9
                )

            plt.ylabel(metric, fontsize=20)
            plt.title(f"{metric} by Race", fontsize=20, fontweight='bold')
            plt.xticks(
                x + bar_width * (n_models - 1) / 2,
                races,
                rotation=45,
                ha="right",
                fontsize=25, # Increased race name font size
            )
            plt.yticks(fontsize=20)

            plt.legend(
                title="Model",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.5), # Adjusted position for taller figure
                ncol=min(len(models), 4),
                fontsize=18,
                title_fontsize=14
            )

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout() # Let savefig with bbox_inches handle final fit

            # Sanitize metric name for the output filename
            metric_clean = re.sub(r"[^a-zA-Z0-9]+", "", metric)
            out_path = os.path.join(folder_path, f"race_{metric_clean}_comparison.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {metric} plot to {out_path}")

# Example usage:
# Make sure the directory "Toxicity_results/debias prompt" exists and contains
# the necessary CSV files before running.
plot_race_barplots("Toxicity_results/toxic prompt")

