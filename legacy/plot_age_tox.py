import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import re

# Directory containing the CSV files
directory_path = "Toxicity_results/debias prompt/threshold_0.2"

# Define the order of age groups
age_order = ["[0-20]", "[20-40]", "[40-65]", "[65+]", "unknown"]

# Okabe-Ito color palette
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#999999"
]

# List all CSV files ending with age_metrics.csv
csv_files = [f for f in os.listdir(directory_path) if f.endswith("age_metrics.csv")]

# Container for all data
all_data = []

print("Loading CSV files...")
for file in tqdm(csv_files, desc="Processing files"):
    # Extract clean model name
    model_name = re.sub(
        r"[_-](temprature|temperature|nucleus|maxlen|gender_metrics).*",
        "",
        file.replace("_age_metrics.csv", ""),
        flags=re.IGNORECASE
    )
    
    # Load CSV
    df = pd.read_csv(os.path.join(directory_path, file))
    df["Model"] = model_name  # Add model name column
    
    # Keep relevant columns
    all_data.append(df[["Age", "Metric", "Mean", "CI_low", "CI_high", "Model"]])

# Combine into one big dataframe
df_all = pd.concat(all_data, ignore_index=True)

# Ensure consistent ordering of age groups
df_all["Age"] = pd.Categorical(df_all["Age"], categories=age_order, ordered=True)

# Get consistent model ordering
models = sorted(df_all["Model"].unique())

# Assign consistent colors to models
color_cycle = itertools.cycle(OKABE_ITO)
model_colors = {model: next(color_cycle) for model in models}

# Get unique metrics
metrics = df_all["Metric"].unique()

print("Creating plots...")
for metric in tqdm(metrics, desc="Plotting metrics"):
    df_metric = df_all[df_all["Metric"] == metric]
    
    # Pivot data into multi-index: (Age, Model)
    grouped = df_metric.set_index(["Age", "Model"])[["Mean", "CI_low", "CI_high"]].sort_index()
    
    # Reindex for consistent order
    grouped = grouped.reindex(pd.MultiIndex.from_product([age_order, models], names=["Age", "Model"]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bar_width = 0.8 / len(models)  # distribute bars within each age group
    x = range(len(age_order))
    
    for i, model in enumerate(models):
        model_data = grouped.xs(model, level="Model")
        means = model_data["Mean"]
        ci_low = model_data["CI_low"]
        ci_high = model_data["CI_high"]
        
        # Compute asymmetric error bars
        yerr = [means - ci_low, ci_high - means]
        
        ax.bar(
            [pos + i * bar_width for pos in x],
            means,
            width=bar_width,
            label=model,
            yerr=yerr,
            capsize=3,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9
        )
    
    ax.set_title(f"{metric} across Age Groups", fontsize=20, fontweight="bold")
    ax.set_xlabel("Age Group", fontsize=20)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xticks([pos + bar_width * (len(models) / 2) for pos in x])
    ax.set_xticklabels(age_order, fontsize=20)
    
    # Legend placed UNDER the plot
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),  # shift down
        ncol=3,  # adjust number of columns depending on model count
        fontsize=19,
        title_fontsize=14
    )
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(directory_path, f"age_{metric.replace(' ', '_')}_comparison.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

print("✅ All age comparison plots saved with legend neatly placed under the plot.")
