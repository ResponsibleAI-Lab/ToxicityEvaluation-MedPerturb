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

def plot_gender_barplots(directory_path):
    """
    Generates and saves bar plots comparing model metrics across different genders.
    This function is robust and can handle a path to a parent directory containing
    subfolders, or a path to a single, specific folder.
    """
    GENDER_ORDER = ["Male", "Female", "Unknown"]

    # --- Robust path handling ---
    subfolders = [os.path.join(directory_path, name) for name in os.listdir(directory_path)
                  if os.path.isdir(os.path.join(directory_path, name))]

    if not subfolders:
        if any(f.endswith("gender_metrics.csv") for f in os.listdir(directory_path)):
            folders_to_process = [directory_path]
        else:
            folders_to_process = []
    else:
        folders_to_process = subfolders

    if not folders_to_process:
        print(f"Error: No subfolders or 'gender_metrics.csv' files found in '{directory_path}'.")
        return
    # --- End of path handling ---

    for folder_path in tqdm(folders_to_process, desc="Processing folders"):
        subfolder_name = os.path.basename(folder_path)
        print(f"\n📂 Processing folder: {folder_path}")
        
        csv_files = [f for f in os.listdir(folder_path) if f.endswith("gender_metrics.csv")]
        all_data = []

        for file in tqdm(csv_files, desc=f"Loading CSVs in {subfolder_name}"):
            path = os.path.join(folder_path, file)
            df = pd.read_csv(path)
            df['Gender'] = df['Gender'].str.strip().str.title()
            
            model_name = re.sub(
                r"[_-](temprature|temperature|nucleus|maxlen|race_metrics).*",
                "",
                file.replace("_gender_metrics.csv", ""),
                flags=re.IGNORECASE
            )
            df["Model"] = model_name
            all_data.append(df)

        if not all_data:
            continue

        plot_df = pd.concat(all_data, ignore_index=True)
        genders = GENDER_ORDER
        models = sorted(plot_df["Model"].unique())
        metrics = plot_df["Metric"].unique()

        for metric in tqdm(metrics, desc=f"Plotting in {subfolder_name}"):
            metric_df = plot_df[plot_df["Metric"] == metric]
            
            n_genders = len(genders)
            n_models = len(models)
            bar_width = 0.8 / n_models
            x = np.arange(n_genders)

            # --- STYLING UPDATES to match race plots ---
            plt.figure(figsize=(14, 10)) # Standardized figure size
            for i, model in enumerate(models):
                model_data = metric_df[metric_df["Model"] == model].set_index("Gender").reindex(genders)
                means = model_data["Mean"].values
                ci_low = model_data["CI_low"].values
                ci_high = model_data["CI_high"].values

                lower = np.clip(means - ci_low, 0, None)
                upper = np.clip(ci_high - means, 0, None)
                yerr = [lower, upper]

                plt.bar(
                    x + i * bar_width, means, width=bar_width, yerr=yerr,
                    capsize=4, label=model, color=OKABE_ITO[i % len(OKABE_ITO)], alpha=0.9
                )

            plt.ylabel(metric, fontsize=14, fontweight='bold')
            plt.title(f"{metric} by Gender", fontsize=16, fontweight='bold')
            plt.xticks(
                x + bar_width * (n_models - 1) / 2, genders,
                rotation=45, ha="right", fontsize=20
            )
            plt.yticks(fontsize=20)

            # --- MODIFICATION: Adjusted legend position ---
            plt.legend(
                title="Model", loc="upper center",
                bbox_to_anchor=(0.5, -0.5), # MOVED legend slightly up
                ncol=min(len(models), 4), fontsize=18, title_fontsize=14
            )
            # --- END OF MODIFICATION ---
            
            # --- END OF STYLING UPDATES ---

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            metric_clean = re.sub(r"[^a-zA-Z0-9]+", "", metric)
            out_path = os.path.join(folder_path, f"gender_{metric_clean}_comparison.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {metric} plot to {out_path}")

# Example usage:
plot_gender_barplots("Toxicity_results/normal prompt")

