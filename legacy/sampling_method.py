import json
import random

# Function to read the JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to sample 1000 entries
def sample_data(data, num_samples=1000):
    intersentence_data = data['data']['intersentence']
    
    # Ensure the number of samples doesn't exceed the available data
    if num_samples > len(intersentence_data):
        num_samples = len(intersentence_data)
    
    # Randomly sample 1000 entries
    sampled_data = random.sample(intersentence_data, num_samples)
    
    # Create the final output with the sampled data
    sampled_json = {
        "version": data['version'],
        "data": {
            "intersentence": sampled_data
        }
    }
    
    return sampled_json

# Save the sampled data to a new file
def save_sampled_json(sampled_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(sampled_data, file, indent=4)

# Main execution
if __name__ == "__main__":
    # Load the original JSON file
    file_path = './data/dev.json'  # Replace with the path to your JSON file
    data = load_json(file_path)
    
    # Sample 1000 entries
    sampled_data = sample_data(data, num_samples=1000)
    
    # Save the sampled data to a new JSON file
    output_file = './data/sampled_dev.json'  # Output file name
    save_sampled_json(sampled_data, output_file)

    print("Sampled data saved to", output_file)
