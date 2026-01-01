# semantic_filtering.py
# Step-2
# Filtering criterion: score > threshold

import json
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BGE Reranker path
BGE_RERANKER = "bge-reranker-v2-m3" # Use Hugging Face model name or local path

# Path settings
BASE_DIR = Path("input Low Intensity Factoid Conflict Dataset Folder Path")
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "Chunk"  # Input from chunk.py output
OUTPUT_DIR = DATA_DIR / "Semantic_Filtered"  # Output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Model
# ============================================================
def load_model():
    """Load BGE Reranker model and tokenizer."""
    print(f"Loading Reranker: {BGE_RERANKER}")
    tokenizer = AutoTokenizer.from_pretrained(BGE_RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANKER)
    model.to(DEVICE).eval()
    return tokenizer, model


# ============================================================
# Compute Semantic Scores
# ============================================================
@torch.no_grad()
def compute_semantic_scores(tokenizer, model, pairs, batch_size=16):

    q = [p["sentence_1"]["text"] for p in pairs]
    d = [p["sentence_2"]["text"] for p in pairs]

    scores = []

    for start in tqdm(range(0, len(pairs), batch_size), ncols=120):
        end = min(start + batch_size, len(pairs))

        inputs = tokenizer(
            q[start:end],
            d[start:end],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        logits = model(**inputs).logits.squeeze(-1).cpu().tolist()
        scores.extend(logits)

    return scores


# ============================================================
# Gap-based Threshold Filtering
# ============================================================
def gap_threshold(values):
    """Find threshold based on the largest drop in score distribution."""
    sorted_vals = sorted(values, reverse=True)
    diffs = np.diff(sorted_vals)
    idx = np.argmin(diffs)
    return (sorted_vals[idx] + sorted_vals[idx + 1]) / 2


# ============================================================
# Process Single File
# ============================================================
def process_file(input_path, output_path, tokenizer, model):
    """Process a single event file for semantic filtering."""
    print("\nProcessing:", input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Compute semantic similarity
    semantic_scores = compute_semantic_scores(tokenizer, model, pairs)

    # Determine threshold
    th = gap_threshold(semantic_scores)
    print("Semantic threshold:", th)

    # Update JSON with scores and filtering results
    for p, score in zip(pairs, semantic_scores):
        p["semantic_score"] = score
        p["semantic_filtered"] = 1 if score > th else 0

    # Save to OUTPUT directory
    with open(output_path, "w", encoding="utf-8") as g:
        json.dump(pairs, g, ensure_ascii=False, indent=2)

    print("Saved:", output_path)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    tokenizer, model = load_model()

    # Get all JSON files from chunk output directory
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])

    for fname in files:
        input_path = INPUT_DIR / fname
        output_path = OUTPUT_DIR / fname
        process_file(input_path, output_path, tokenizer, model)

    print("\nProcessing complete.")