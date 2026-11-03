import os
import pandas as pd
import re

def calculate_mean_metrics_by_race(directory_path):
    """
    Calculates the mean value for each metric, grouped by race, across all
    models in a directory.

    Args:
        directory_path (str): The path to the directory containing the CSV files.

    Returns:
        pandas.DataFrame: A DataFrame with races as the index, metrics as columns,
                          and the calculated mean values. Returns an empty DataFrame
                          if no valid files are found.
    """
    all_data = []

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' was not found.")
        return pd.DataFrame()

    # Find all files ending with 'race_metrics.csv'
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith("race_metrics.csv")]
        if not files:
            print(f"Warning: No files ending with 'race_metrics.csv' found in '{directory_path}'.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
        return pd.DataFrame()

    print(f"Found {len(files)} metric files to process...")

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            df = pd.read_csv(file_path)

            if not df.empty:
                # Normalize race labels: remove whitespace and capitalize
                df['Race'] = df['Race'].str.strip().str.title()
                # Normalize metric names
                df['Metric'] = df['Metric'].str.strip()
                all_data.append(df)
            else:
                print(f"  - No data found in {filename}")
        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")


    if not all_data:
        print("Could not find any data in the readable files.")
        return pd.DataFrame()

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by Metric and Race, then calculate the mean of the 'Mean' column
    mean_metrics = combined_df.groupby(['Metric', 'Race'])['Mean'].mean()

    # Unstack the 'Metric' level to turn it into columns for a pivot-table-like view
    result_df = mean_metrics.unstack(level='Metric')

    return result_df

if __name__ == "__main__":
    # --- HARDCODED DIRECTORY ---
    # Replace this path with the actual path to your data directory.
    data_directory = "Toxicity_results/normal prompt/threshold_0.2"
    
    print(f"\n--- Calculating Mean Metrics for directory: '{data_directory}' ---")
    mean_metric_results = calculate_mean_metrics_by_race(data_directory)

    if not mean_metric_results.empty:
        print("\n--- Average Metric Values Across All Models ---")
        # Sort the results alphabetically by race for consistent output
        print(mean_metric_results.sort_index().to_string())
        print("---------------------------------------------")

