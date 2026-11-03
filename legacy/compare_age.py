import pandas as pd
import os
import glob

def load_csv_files(directory_path):
    """Load all CSV files from directory into a dictionary of DataFrames"""
    csv_files = glob.glob(os.path.join(directory_path, "*age_metrics.csv"))
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
    """Calculate percentage of age groups that improved for each metric"""
    results = {}
    
    for metric in metrics:
        improved_count = 0
        total_age_groups = 0
        
        for age_group in original_df['Age'].unique():
            if age_group not in debiased_df['Age'].values:
                continue
                
            orig_data = original_df[(original_df['Age'] == age_group) & (original_df['Metric'] == metric)]
            debiased_data = debiased_df[(debiased_df['Age'] == age_group) & (debiased_df['Metric'] == metric)]
            
            if len(orig_data) > 0 and len(debiased_data) > 0:
                total_age_groups += 1
                orig_mean = orig_data['Mean'].values[0]
                debiased_mean = debiased_data['Mean'].values[0]
                
                # Improvement means reduction in toxicity (lower value is better)
                if debiased_mean < orig_mean:
                    improved_count += 1
        
        # Calculate percentage of age groups that improved
        if total_age_groups > 0:
            improvement_pct = (improved_count / total_age_groups) * 100
        else:
            improvement_pct = 0
            
        results[metric] = {
            'improved_groups': improved_count,
            'total_groups': total_age_groups,
            'improvement_percentage': improvement_pct
        }
    
    return results

def main():
    # Paths to your directories
    original_dir = input("Enter path to original directory: ").strip()
    debiased_dir = input("Enter path to debiased directory: ").strip()
    
    # Load CSV files from both directories
    original_data = load_csv_files(original_dir)
    debiased_data = load_csv_files(debiased_dir)
    
    # Define the metrics we're interested in
    metrics = ['EMT', 'Toxicity Fraction', 'EPBG']
    
    # Compare files with the same name
    comparison_results = {}
    
    for filename in original_data:
        if filename in debiased_data:
            print(f"\nComparing: {filename}")
            print("-" * 50)
            
            orig_df = original_data[filename]
            debiased_df = debiased_data[filename]
            
            # Calculate improvement percentages
            results = calculate_improvement_percentage(orig_df, debiased_df, metrics)
            
            # Store results
            comparison_results[filename] = results
            
            # Print results for this file
            model_name = filename.split('_')[0]
            print(f"Model: {model_name}")
            print("Improvement Results:")
            print("Metric\t\tImproved Groups\tTotal Groups\tImprovement %")
            print("-" * 55)
            
            for metric, data in results.items():
                print(f"{metric:15}\t{data['improved_groups']}/{data['total_groups']}\t\t{data['improvement_percentage']:.1f}%")
            
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
        results_df.to_csv('debiasing_improvement_analysis.csv', index=False)
        print("Results saved to 'debiasing_improvement_analysis.csv'")
        
        # Print summary table
        print("\nSummary Table:")
        summary_df = results_df.pivot_table(
            index='Model', 
            columns='Metric', 
            values='Improvement_Percentage', 
            aggfunc='first'
        )
        print(summary_df.to_string(float_format="%.1f%%"))
    
    return comparison_results

if __name__ == "__main__":
    results = main()