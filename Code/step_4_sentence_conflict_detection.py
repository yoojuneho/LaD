# sentence_conflict_detection.py
# Step-4: Conflict Detection using GPT
# Only verify conflict candidates (semantic_filtered=0 AND cosine_filtered=0) with GPT

import json
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# CONFIG
# ==========================================
client = OpenAI(api_key="OPENAI_API_KEY")  # Replace with your API key

# Path settings - modify these according to your environment
BASE_DIR = Path("./")  # Project root directory
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "chunk_output_cosine"  # Input from cosine.py output
OUTPUT_DIR = DATA_DIR / "sentence_detected"  # Output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "gpt-4o"

# ==========================================
# GPT Prompt
# ==========================================
def build_prompt_zero_shot(s1, s2):
    return f"""You are a conflict detector for sentence pairs.

TASK: Decide if the two sentences make mutually incompatible claims about the same single fact.

CONFLICT = 1 when:
- For the same event/entity/measure, they give different numbers/dates/times/locations/agents; or
- One asserts occurrence while the other denies it; or
- Core outcome for the same event is incompatible (e.g., winner/responsible party/casualty count).

NO CONFLICT = 0 when:
- They discuss different facts/events or different scopes (subset/superset) without contradiction;
- Quantifiers/hedges explain the difference (“at least/around/at most/estimated/reported” vs an exact value);
- A plausible timepoint/update difference explains it;
- Modal/hedged statements don’t directly contradict a categorical claim.

Rules:
- Use only the two sentences; no outside knowledge.
- Normalize numbers/dates/units; treat synonyms/pronouns as the same referent when clear.
- If uncertain they refer to the same fact, output 0.
- Output only one character with no explanation: 1 for conflict, 0 otherwise.

Sentence A: "{s1}"
Sentence B: "{s2}"

Answer:"""

# ==========================================
# GPT Request Function
# ==========================================
def ask_gpt_for_conflict(s1, s2):
    """
    Request GPT to determine if two sentences conflict.
    
    Args:
        s1: First sentence text
        s2: Second sentence text
    
    Returns:
        int: 1 if conflict detected, 0 otherwise
    """
    prompt = build_prompt_zero_shot(s1, s2)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1
        )
        answer = response.choices[0].message.content.strip()
        return 1 if answer == "1" else 0

    except Exception as e:
        print(f"GPT API Error: {e}")
        return 0


# ==========================================
# Process Event File
# ==========================================
def process_event_file(input_path, output_path):
    """
    Process a single event file for GPT conflict detection.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    
    Returns:
        int: Number of conflicts detected
    """
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Reset labels
    for p in pairs:
        p["conflict_label"] = 0

    # Filter GPT target pairs (survived both semantic and cosine filtering)
    gpt_targets = [
        p for p in pairs
        if p.get("semantic_filtered") == 0 and p.get("cosine_filtered") == 0
    ]

    # Perform GPT detection
    conflicts_detected = 0

    print(f"Processing {len(gpt_targets)} candidate pairs with GPT...")
    
    for p in tqdm(gpt_targets, desc=f"GPT Detection ({input_path.name})", ncols=120):
        s1 = p["sentence_1"]["text"]
        s2 = p["sentence_2"]["text"]

        label = ask_gpt_for_conflict(s1, s2)
        p["conflict_label"] = label

        if label == 1:
            conflicts_detected += 1

    # Save file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    return conflicts_detected


# ==========================================
# Extract Event ID for Sorting
# ==========================================
def extract_event_id(filename):
    """Extract event ID from filename for sorting."""
    try:
        return int(filename.stem.replace("event_", ""))
    except:
        return 999999999


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # Get all JSON files from cosine filtered directory
    files = sorted(
        [f for f in INPUT_DIR.iterdir() if f.suffix == ".json"],
        key=lambda x: extract_event_id(x)
    )
    
    if not files:
        print(f"No JSON files found in {INPUT_DIR}")
        exit(1)

    print(f"Found {len(files)} event files")

    total_conflicts = 0

    for input_file in files:
        output_file = OUTPUT_DIR / input_file.name
        conflicts_detected = process_event_file(input_file, output_file)
        total_conflicts += conflicts_detected

        print(f"Event {input_file.name}: {conflicts_detected} conflicts detected")

    # =============================
    # Final Statistics
    # =============================
    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL REPORT")
    print("=" * 60)
    print(f"Total events processed: {len(files)}")
    print(f"Total conflicts detected: {total_conflicts}")
    print("=" * 60)