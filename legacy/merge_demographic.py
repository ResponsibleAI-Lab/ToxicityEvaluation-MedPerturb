import json
import argparse

def add_demographics_inplace(demo_file, target_file):
    # Load both files
    with open(demo_file, "r") as f:
        demo_data = json.load(f)

    with open(target_file, "r") as f:
        target_data = json.load(f)

    # Check if lengths match
    if len(demo_data) != len(target_data):
        print("❌ Files have different lengths. Aborting update.")
        return

    # Verify all dataset_id and context_id match
    for i, (demo_item, target_item) in enumerate(zip(demo_data, target_data)):
        if (
            demo_item.get("dataset_id") != target_item.get("dataset_id")
            or demo_item.get("context_id") != target_item.get("context_id")
        ):
            print(
                f"❌ Mismatch at index {i}: "
                f"demo({demo_item.get('dataset_id')}, {demo_item.get('context_id')}) vs "
                f"target({target_item.get('dataset_id')}, {target_item.get('context_id')})"
            )
            print("Aborting update. No changes were made.")
            return

    # If all match, perform the update
    for demo_item, target_item in zip(demo_data, target_data):
        for field in ["age", "gender", "race"]:
            if field in demo_item:
                target_item[field] = demo_item[field]

    # Save back to the target file
    with open(target_file, "w") as f:
        json.dump(target_data, f, indent=2)

    print(f"✅ All sample IDs matched. Target file updated: {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add demographics from one JSON into another JSON in place, only if all IDs match")
    parser.add_argument("--demo_file", type=str, required=True, help="Path to the demographics JSON file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to the target JSON file to be updated")
    args = parser.parse_args()

    add_demographics_inplace(args.demo_file, args.target_file)
