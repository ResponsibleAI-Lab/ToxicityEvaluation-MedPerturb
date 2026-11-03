import pandas as pd
import os
import glob

def load_csv_files(directory_path, file_pattern):
    """Load all CSV files from directory into a dictionary of DataFrames"""
    csv_files = glob.glob(os.path.join(directory_path, file_pattern))
    dataframes = {}
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            dataframes[filename] = df
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return dataframes

def calculate_improvement_percentage(original_df, debiased_df, metrics):
    """Calculate percentage of demographic groups that improved for each metric"""
    results = {}
    
    for metric in metrics:
        improved_count = 0
        total_groups = 0
        
        for group in original_df['Race'].unique():
            if group not in debiased_df['Race'].values:
                continue
                
            orig_data = original_df[(original_df['Race'] == group) & (original_df['Metric'] == metric)]
            debiased_data = debiased_df[(debiased_df['Race'] == group) & (debiased_df['Metric'] == metric)]
            
            if len(orig_data) > 0 and len(debiased_data) > 0:
                total_groups += 1
                orig_mean = orig_data['Mean'].values[0]
                debiased_mean = debiased_data['Mean'].values[0]
                
                # Improvement means reduction in toxicity (lower value is better)
                if debiased_mean < orig_mean:
                    improved_count += 1
        
        # Calculate percentage of groups that improved
        if total_groups > 0:
            improvement_pct = (improved_count / total_groups) * 100
        else:
            improvement_pct = 0
            
        results[metric] = {
            'improved_groups': improved_count,
            'total_groups': total_groups,
            'improvement_percentage': improvement_pct
        }
    
    return results

def main():
    # Hardcoded directory paths - replace with your actual paths
    original_dir = "Toxicity_results/toxic prompt/threshold_0.2"  # Replace with your actual path
    debiased_dir = "Toxicity_results/normal prompt/threshold_0.2"  # Replace with your actual path
    
    # File pattern for race metrics
    file_pattern = "*_race_metrics.csv"
    
    # Load CSV files from both directories
    print(f"Loading files from original directory: {original_dir}")
    original_data = load_csv_files(original_dir, file_pattern)
    
    print(f"\nLoading files from debiased directory: {debiased_dir}")
    debiased_data = load_csv_files(debiased_dir, file_pattern)
    
    # Define the metrics we're interested in
    metrics = ['EMT', 'Toxicity Fraction', 'EPBG']
    
    # Compare files with the same name
    comparison_results = {}
    
    print("\n" + "="*60)
    print("RACE DEBIASING IMPROVEMENT ANALYSIS")
    print("="*60)
    
    for filename in original_data:
        if filename in debiased_data:
            print(f"\nComparing: {filename}")
            print("-" * 50)
            
            orig_df = original_data[filename]
            debiased_df = debiased_data[filename]
            
            # Print the race groups found in this file for verification
            print(f"Race groups found: {sorted(orig_df['Race'].unique())}")
            
            # Calculate improvement percentages
            results = calculate_improvement_percentage(orig_df, debiased_df, metrics)
            
            # Store results
            comparison_results[filename] = results
            
            # Print results for this file
            model_name = filename.split('_')[0]
            print(f"Model: {model_name}")
            print("Race Groups Improvement Results:")
            print("Metric\t\t\tImproved Groups\tTotal Groups\tImprovement %")
            print("-" * 65)
            
            for metric, data in results.items():
                print(f"{metric:20}\t{data['improved_groups']}/{data['total_groups']}\t\t{data['improvement_percentage']:.1f}%")
            
            print()
    
    # Save detailed results to CSV
    if comparison_results:
        all_results = []
        for filename, metrics_data in comparison_results.items():
            model_name = filename.split('_')[0]
            for metric, data in metrics_data.items():
                all_results.append({
                    'Model': model_name,
                    'Filename': filename,
                    'Metric': metric,
                    'Improved_Groups': data['improved_groups'],
                    'Total_Groups': data['total_groups'],
                    'Improvement_Percentage': data['improvement_percentage']
                })
        
        results_df = pd.DataFrame(all_results)
        results_csv_path = 'race_debiasing_improvement_analysis.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to '{results_csv_path}'")
        
        # Print summary table
        print("\n" + "="*50)
        print("SUMMARY TABLE: RACE DEBIASING EFFECTIVENESS")
        print("="*50)
        summary_df = results_df.pivot_table(
            index='Model', 
            columns='Metric', 
            values='Improvement_Percentage', 
            aggfunc='first'
        )
        print(summary_df.to_string(float_format="%.1f%%"))
        
        # Calculate overall averages
        print("\n" + "="*50)
        print("OVERALL AVERAGE IMPROVEMENT BY METRIC")
        print("="*50)
        avg_improvement = results_df.groupby('Metric')['Improvement_Percentage'].mean()
        for metric, avg in avg_improvement.items():
            print(f"{metric:20}: {avg:.1f}%")
        
        # Calculate overall averages by model
        print("\n" + "="*50)
        print("OVERALL AVERAGE IMPROVEMENT BY MODEL")
        print("="*50)
        model_avg_improvement = results_df.groupby('Model')['Improvement_Percentage'].mean()
        for model, avg in model_avg_improvement.items():
            print(f"{model:20}: {avg:.1f}%")
    
    else:
        print("No matching files found between the two directories.")
    
    return comparison_results

if __name__ == "__main__":
    # Replace these paths with your actual directory paths
    original_dir = "Toxicity_results/normal prompt/threshold_0.2"  # Replace with your actual path
    debiased_dir = "Toxicity_results/toxic prompt/threshold_0.2"  # Replace with your actual path
    
    results = main()