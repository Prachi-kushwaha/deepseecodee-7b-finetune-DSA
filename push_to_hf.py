from datasets import load_dataset

# Load your JSONL files
ds = load_dataset(
    "json",
    data_files={
        "explanation": "hf_dataset/explanation.jsonl",
        "code": "hf_dataset/code.jsonl",
        "mcq": "hf_dataset/mcq.jsonl",
    }
)

# Push to Hugging Face Hub
ds.push_to_hub("Prachi01/dsa-dataset")