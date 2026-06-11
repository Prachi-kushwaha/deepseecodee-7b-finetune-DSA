
"""
DSA Fine-tuning Dataset Generator
Teacher model : Gemma4 (via Ollama)
Output format : Alpaca JSONL  →  HuggingFace TRL / SFTTrainer
"""

import json
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import argparse
from pathlib import Path
from topics import DSA_TOPICS


OUTPUT_FILE  = "dataset.jsonl"
DELAY_SEC    = 1.0
MAX_RETRIES  = 3
MAX_EXAMPLES = 150

load_dotenv()

# client = ChatOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

MODEL_NAME = "gpt-4.1-mini"
# # Prompt templates (teacher → student distillation style)


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.7,
    max_tokens=30000,
    timeout=60
)

def concept_prompt(topic: str, category: str, difficulty: str) -> dict:
    """Returns an Alpaca record for concept explanation."""
    instruction = f"Explain the DSA concept: {topic}"
    input_ctx   = f"Category: {category} | Difficulty: {difficulty}"
    system      = (
        "You are an expert DSA teacher. Give a thorough explanation including:\n"
        "1. What the concept is (definition)\n"
        "2. Why it is important / when to use it\n"
        "3. Step-by-step working with a small example\n"
        "4. Time and space complexity\n"
        "5. Common mistakes to avoid\n"
        "Be precise and educational."
    )
    return {"system": system, "instruction": instruction, "input": input_ctx}


def coding_prompt(topic: str, category: str, difficulty: str) -> dict:
    """Returns an Alpaca record for a coding problem."""
    instruction = f"Write a well-commented Python solution for: {topic}"
    input_ctx   = f"Category: {category} | Difficulty: {difficulty}"
    system      = (
        "You are an expert competitive programmer. Provide:\n"
        "1. Problem statement (2-3 lines)\n"
        "2. Approach / algorithm explanation for brute and optimized version if multiple optimized code available provide all\n"
        "3. Clean, well-commented Python code for both brute force and optimized version if multiple optimized code available provide all\n"
        "4. Time complexity and space complexity analysis\n"
        "5. 2 to 3 example with input and output with different different scenarios\n"
        "Use only standard Python. Do not import external libraries."
    )
    return {"system": system, "instruction": instruction, "input": input_ctx}


def mcq_prompt(topic: str, category: str, difficulty: str) -> dict:
    """Returns an Alpaca record for an MCQ question."""
    instruction = f"Create a multiple-choice question about: {topic}"
    input_ctx   = f"Category: {category} | Difficulty: {difficulty}"
    system      = (
        "You are a DSA quiz creator. Generate exactly ONE MCQ with:\n"
        "- A clear question\n"
        "- Four options labeled A, B, C, D\n"
        "- The correct answer clearly stated\n"
        "- A brief explanation of why the answer is correct\n"
        "Format strictly as:\n"
        "Question: ...\n"
        "A) ...\n"
        "B) ...\n"
        "C) ...\n"
        "D) ...\n"
        "Answer: <letter>\n"
        "Explanation: ..."
    )
    return {"system": system, "instruction": instruction, "input": input_ctx}


PROMPT_BUILDERS = [concept_prompt, coding_prompt, mcq_prompt]
PROMPT_TYPES    = ["concept", "coding", "mcq"]


def call_openai(system: str, instruction: str, input_ctx: str) -> str | None:

    prompt = f"""
Instruction:
{instruction}

Input:
{input_ctx}
"""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt)
            ])

            return response.content.strip()

        except Exception as e:
            print(f"⚠ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2 ** attempt)

    return None


def build_alpaca_record(
    topic: str,
    category: str,
    difficulty: str,
    prompt_type: str,
    instruction: str,
    input_ctx: str,
    output: str,
) -> dict:
    """Build a single Alpaca-format JSONL record with metadata."""
    return {
        "instruction": instruction,
        "input": input_ctx,
        "output": output,
        "metadata": {
            "topic": topic,
            "category": category,
            "difficulty": difficulty,
            "type": prompt_type,
            "model": MODEL_NAME,
        }
    }


def generate_dataset(
    output_file: str = OUTPUT_FILE,
    start_index: int = 0,
    only_type: str | None = None,     # "concept" | "coding" | "mcq" | None (all)
):
    out_path = Path(output_file)
    # Count existing lines so we can resume
    existing = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            existing = sum(1 for _ in f)
        print(f"Resuming — {existing} records already in {output_file}")

    skipped   = 0
    generated = 0
    failed    = 0

    topics_to_run = DSA_TOPICS[start_index:]
    total = len(topics_to_run) * (
        len(PROMPT_BUILDERS) if not only_type
        else 1
    )
    print(f" Generating ~{total} records for {len(topics_to_run)} topics...\n")

    with open(out_path, "a", encoding="utf-8") as fout:
        for idx, entry in enumerate(topics_to_run, start=start_index):
            topic      = entry["topic"]
            category   = entry["category"]
            difficulty = entry["difficulty"]

            builders = list(zip(PROMPT_TYPES, PROMPT_BUILDERS))
            if only_type:
                builders = [(t, b) for t, b in builders if t == only_type]

            for ptype, builder in builders:
                if generated >= MAX_EXAMPLES:
                    print("\nReached max example limit.")
                    return
                global_idx = idx * len(PROMPT_BUILDERS) + PROMPT_TYPES.index(ptype)

                # Skip already-written records (resume support)
                if global_idx < existing:
                    skipped += 1
                    continue

                print(f"  [{generated + failed + 1}/{total}] {ptype:8s} | {topic}")

                tmpl   = builder(topic, category, difficulty)
                output = call_openai(
                    tmpl["system"], tmpl["instruction"], tmpl["input"]
                )

                if output:
                    record = build_alpaca_record(
                        topic, category, difficulty,
                        ptype,
                        tmpl["instruction"],
                        tmpl["input"],
                        output,
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    generated += 1
                else:
                    print(f" x FAILED — skipping this record")
                    failed += 1

                time.sleep(DELAY_SEC)

    print(f"\n Done! Generated: {generated} | Failed: {failed} | Skipped: {skipped}")
    print(f" Output saved to: {out_path.resolve()}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DSA fine-tuning dataset using Gemma4")
    parser.add_argument("--output",type=str, default=OUTPUT_FILE,help="Output JSONL file path")
    parser.add_argument("--start", type=int, default=0, help="Start from topic index (for resuming)")
    parser.add_argument("--type", type=str, default=None, help="Only generate: concept | coding | mcq")
    args = parser.parse_args()

    generate_dataset(
        output_file=args.output,
        start_index=args.start,
        only_type=args.type,
    )