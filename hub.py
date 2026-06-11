"""
convert_to_hf.py
----------------
Converts your DSA dataset (JSON / JSONL) into HuggingFace-ready JSONL files,
split by type: explanation, code, mcq.

Usage:
    python convert_to_hf.py --input data.json --output_dir hf_dataset

Then push with:
    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "explanation": "hf_dataset/explanation.jsonl",
        "code":        "hf_dataset/code.jsonl",
        "mcq":         "hf_dataset/mcq.jsonl",
    })
    ds.push_to_hub("your-username/dsa-dataset")
"""

import json
import os
import re
import argparse
from pathlib import Path


def extract_code_blocks(text: str) -> str:
    """Pull out only the fenced code blocks from a markdown string."""
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(b.strip() for b in blocks) if blocks else text.strip()


def extract_mcq_blocks(text: str) -> str:
    """Return MCQ content as-is."""
    return text.strip()

def build_record(raw: dict, rec_type: str, idx: int) -> dict:
    """
    Build one HF-ready record from a raw JSON object.

    rec_type: "explanation" | "code" | "mcq"
    """
    topic      = raw.get("metadata", {}).get("topic", "unknown")
    category   = raw.get("metadata", {}).get("category", "")
    difficulty = raw.get("metadata", {}).get("difficulty", "")
    output     = raw.get("output", "")

    # craft a type-specific prompt & response
    if rec_type == "explanation":
        prompt   = f"Explain the DSA concept: {topic}. Category: {category}, Difficulty: {difficulty}"
        response = output  # keep full explanation as-is

    elif rec_type == "code":
        prompt   = f"Write code to implement: {topic}. Category: {category}, Difficulty: {difficulty}"
        response = extract_code_blocks(output)

    elif rec_type == "mcq":
        prompt   = f"Generate an MCQ question for the DSA concept: {topic}. Category: {category}, Difficulty: {difficulty}"
        response = extract_mcq_blocks(output)

    else:
        raise ValueError(f"Unknown type: {rec_type}")

    return {
        "id":         f"{topic.replace(' ', '-')}-{rec_type}-{idx:04d}",
        "topic":      topic,
        "category":   category,
        "difficulty": difficulty,
        "type":       rec_type,
        "prompt":     prompt,
        "response":   response,
        "source":     raw.get("metadata", {}).get("model", "unknown"),
    }


# Main

def load_input(path: str) -> list[dict]:
    """Load a .json (list or single object) or .jsonl file."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        records = []

        for line_no, line in enumerate(text.splitlines(), 1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {line_no}: {e}")

        return records

def convert(input_path: str, output_dir: str, types: list[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_records = load_input(input_path)
    print(f"Loaded {len(raw_records)} raw record(s) from '{input_path}'")

    writers = {t: open(os.path.join(output_dir, f"{t}.jsonl"), "w", encoding="utf-8")
               for t in types}
    counts  = {t: 0 for t in types}
    TYPE_MAP = {
    "concept": "explanation",
    "coding": "code",
    "mcq": "mcq"
    }

    for idx, raw in enumerate(raw_records):
        raw_type = raw.get("metadata", {}).get("type", "").lower()

        mapped_type = TYPE_MAP.get(raw_type)
        for rec_type in types:
            # Skip if the raw record already has an explicit type that doesn't match
            if mapped_type and mapped_type != rec_type:
               continue

            record = build_record(raw, rec_type, idx)
            writers[rec_type].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[rec_type] += 1

    for w in writers.values():
        w.close()

    print("\n Done! Output files:")
    for t in types:
        fpath = os.path.join(output_dir, f"{t}.jsonl")
        print(f"   {fpath}  →  {counts[t]} record(s)")

    # Print HF push snippet
    print("\n Push to HuggingFace")
    files_arg = ", ".join(
        f'"{t}": "{os.path.join(output_dir, t+".jsonl")}"' for t in types
    )
    print(f"""
from datasets import load_dataset
ds = load_dataset("json", data_files={{ {files_arg} }})
ds.push_to_hub("your-username/dsa-dataset")
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DSA JSON → HF-ready JSONL")
    parser.add_argument("--input",      required=True,  help="Path to input .json or .jsonl file")
    parser.add_argument("--output_dir", default="hf_dataset", help="Output directory (default: hf_dataset)")
    parser.add_argument("--types",      nargs="+", default=["explanation", "code", "mcq"],
                        help="Which splits to generate (default: explanation code mcq)")
    args = parser.parse_args()

    convert(args.input, args.output_dir, args.types)