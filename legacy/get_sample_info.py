from google import genai
from pydantic import BaseModel
import argparse
import json
from tqdm import tqdm
import time
import os

# Define the schema for the expected API response
class Demographics(BaseModel):
    age: str
    gender: str
    race: str

# Initialize the client once to avoid reconnect overhead
client = genai.Client()

# Function to get demographics from the Gemini model
def get_demographics(text: str) -> Demographics:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Extract the age, gender, and race from the following text. "
                 f"If any are not mentioned, respond with 'Unknown'. "
                 f"Text: {text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": Demographics,
        },
    )
    return response.parsed

def annotate_file(prediction_file: str):
    # Load the original data
    with open(prediction_file, "r") as infile:
        original_data = json.load(infile)

    output_file = prediction_file.replace(".json", "_demographics.json")

    # If results already exist, resume from them
    if os.path.exists(output_file):
        with open(output_file, "r") as infile:
            processed_data = json.load(infile)
        print(f"Resuming from existing file: {output_file}")
    else:
        processed_data = original_data

    # Identify starting point: find first record without demographics
    start_idx = 0
    for i, item in enumerate(processed_data):
        if not all(k in item for k in ["age", "gender", "race"]):
            start_idx = i
            break
    else:
        print("All records already processed.")
        return

    # Process only remaining records
    for idx in tqdm(range(start_idx, len(processed_data)), desc="Processing records"):
        item = processed_data[idx]
        context = item.get("clinical_context", "")
        demo = get_demographics(context)

        item.update({
            "age": getattr(demo, "age", None),
            "gender": getattr(demo, "gender", None),
            "race": getattr(demo, "race", None)
        })

        # Save after each iteration
        with open(output_file, "w") as outfile:
            json.dump(processed_data, outfile, indent=2)

        # Respect 30 requests/minute → 2 seconds between requests
        time.sleep(10)

    print(f"Processed file saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate JSON file with demographics via Gemini API")
    parser.add_argument("--prediction_file", type=str, required=True, help="Path to the prediction file")
    args = parser.parse_args()

    annotate_file(args.prediction_file)
