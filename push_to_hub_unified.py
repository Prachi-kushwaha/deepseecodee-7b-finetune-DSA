from datasets import load_dataset, concatenate_datasets

# Load split files
ds = load_dataset(
    "json",
    data_files={
        "explanation": "hf_dataset/explanation.jsonl",
        "code": "hf_dataset/code.jsonl",
        "mcq": "hf_dataset/mcq.jsonl",
    }
)

# Merge into one dataset
combined = concatenate_datasets([
    ds["explanation"],
    ds["code"],
    ds["mcq"]
])

# Shuffle
combined = combined.shuffle(seed=42)

# Remove unnecessary column
combined = combined.remove_columns(["id"])

# Rename columns for instruction tuning
combined = combined.rename_columns({
    "prompt": "instruction",
    "response": "output"
})

# Push unified dataset
combined.push_to_hub("Prachi01/dsa-dataset-unified")