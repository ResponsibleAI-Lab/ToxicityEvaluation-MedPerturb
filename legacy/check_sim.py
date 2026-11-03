import json

def compare_json_files(file1, file2):
    # Load both files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Check number of samples
    if len(data1) != len(data2):
        print(f"❌ Number of samples differ: {len(data1)} vs {len(data2)}")
        return

    print(f"✅ Both files have {len(data1)} samples.")

    # Compare completions for each sample
    for idx, (s1, s2) in enumerate(zip(data1, data2)):
        completions1 = s1.get("completions", [])
        completions2 = s2.get("completions", [])

        if len(completions1) != len(completions2):
            # print(f"❌ Sample {idx} completions count differ: {len(completions1)} vs {len(completions2)}")
            continue

        # Compare text values
        for c_idx, (c1, c2) in enumerate(zip(completions1, completions2)):
            if c1 != c2:
                # print(f"❌ Difference in sample {idx}, completion {c_idx}")
                # print(f"   File1: {c1[:80]}...")
                # print(f"   File2: {c2[:80]}...")
                break
        else:
            print(f"✅ Sample {idx}: completions match.")

# Example usage
compare_json_files("predictions/MediPhi-Instruct_temprature_1.0_nucleus_1.0_maxlen_300_intrasentence_predictions.json", "predictions/debias prompt/gemma-2-2b-it_temprature_1.0_nucleus_1.0_maxlen_300_intrasentence_predictions.json")
