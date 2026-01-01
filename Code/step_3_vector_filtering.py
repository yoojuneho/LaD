# vector_filtering.py
# Step-3
# Filtering: 1) Auto-filter cosine=1.0, 2) Calculate gap threshold for remaining and filter score < threshold

import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model path
BGE_EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # Use Hugging Face model name or local path

# Path settings (matching chunk.py and bert.py structure)
BASE_DIR = Path("input Low Intensity Factoid Conflict Dataset Folder Path")
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "Semantic_Filtered"  # Input from bert.py output
OUTPUT_DIR = DATA_DIR / "Cosine_Filtered"  # Output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Embedding Model
# ============================================================
def load_embedding_model():
    """Load BGE embedding model for cosine similarity computation."""
    print(f"Loading BGE Embedding Model: {BGE_EMBED_MODEL}")
    model = SentenceTransformer(BGE_EMBED_MODEL)
    model.to(DEVICE)
    return model


# ============================================================
# Compute Cosine Similarity (per event)
# ============================================================
def compute_cosine_similarity_event(model, pairs):
    """
    Compute cosine similarity for all sentence pairs in an event.
    
    Args:
        model: SentenceTransformer model
        pairs: List of sentence pair dictionaries
    
    Returns:
        list: Cosine similarity scores
    """
    sentences1 = [p["sentence_1"]["text"] for p in pairs]
    sentences2 = [p["sentence_2"]["text"] for p in pairs]

    cosine_scores = []
    batch_size = 32

    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))

        emb1 = model.encode(sentences1[start:end], convert_to_tensor=True)
        emb2 = model.encode(sentences2[start:end], convert_to_tensor=True)

        batch_scores = util.cos_sim(emb1, emb2).diagonal().cpu().tolist()
        cosine_scores.extend(batch_scores)

    return cosine_scores


# ============================================================
# Modified Cosine Threshold (excluding 1.0 values)
# ============================================================
def cosine_gap_threshold_excluding_ones(scores):
    """
    Calculate gap-based threshold excluding cosine=1.0 scores.
    (Using the largest gap approach)
    
    Args:
        scores: List of cosine similarity scores
    
    Returns:
        float: Threshold value
    """
    # Exclude 1.0 values (consider 0.9999+ as identical)
    filtered_scores = [s for s in scores if s < 0.9999]

    # If less than 2 scores remain, no meaningful threshold
    if len(filtered_scores) < 2:
        return 0.9999

    # Sort in descending order
    sorted_vals = sorted(filtered_scores, reverse=True)

    # Calculate gaps between adjacent scores
    diffs = np.diff(sorted_vals)

    # Find the index of the largest gap
    idx = np.argmin(diffs)

    # Threshold is the midpoint of this gap
    threshold = (sorted_vals[idx] + sorted_vals[idx + 1]) / 2

    return threshold


# ============================================================
# Extract Event ID from Filename
# ============================================================
def extract_event_id(filename):
    """Extract event ID from filename for sorting."""
    try:
        return int(filename.stem.replace("event_", ""))
    except:
        return 999999999


# ============================================================
# Process Single File
# ============================================================
def process_file(input_path, output_path, model):
    """
    Process a single event file for cosine similarity filtering.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        model: SentenceTransformer model
    
    Returns:
        dict: Statistics about the processing
    """
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # 1) Compute cosine similarity
    cosine_scores = compute_cosine_similarity_event(model, pairs)

    # 2) Calculate modified threshold (excluding 1.0)
    threshold = cosine_gap_threshold_excluding_ones(cosine_scores)
    
    # File-level statistics
    file_perfect = 0
    file_threshold_filtered = 0

    # 3) Apply modified filtering
    for p, score in zip(pairs, cosine_scores):
        p["cosine_score"] = float(score)
        
        # Step 1: Auto-filter if cosine=1.0
        if score >= 0.9999:
            p["cosine_filtered"] = 1
            file_perfect += 1
        # Step 2: Filter if score < threshold
        elif score < threshold:
            p["cosine_filtered"] = 1
            file_threshold_filtered += 1
        # Step 3: Survive if score >= threshold
        else:
            p["cosine_filtered"] = 0

    # 4) Save file
    with open(output_path, "w", encoding="utf-8") as g:
        json.dump(pairs, g, ensure_ascii=False, indent=2)

    return {
        "perfect_matches": file_perfect,
        "threshold_filtered": file_threshold_filtered,
        "threshold": threshold
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    model = load_embedding_model()

    # Get all JSON files from semantic filtered directory
    files = sorted(
        [f for f in INPUT_DIR.iterdir() if f.suffix == ".json"],
        key=lambda x: extract_event_id(x)
    )
    
    if not files:
        print(f"No JSON files found in {INPUT_DIR}")
        exit(1)

    perfect_match_count = 0  # Pairs filtered by cosine≈1.0
    threshold_filtered_count = 0  # Pairs filtered by threshold

    print(f"\nTotal event files: {len(files)}\n")

    pbar = tqdm(files, desc="Processing events", ncols=120)

    for input_file in pbar:
        output_file = OUTPUT_DIR / input_file.name
        
        # Update progress bar with current event
        pbar.set_postfix({
            "event": input_file.stem
        })

        # Process file
        stats = process_file(input_file, output_file, model)
        
        # Update global statistics
        perfect_match_count += stats["perfect_matches"]
        threshold_filtered_count += stats["threshold_filtered"]

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL REPORT")
    print("=" * 60)
    
    print(f"\nFiltering Statistics:")
    print(f"- Perfect matches (cosine≈1.0) filtered: {perfect_match_count} pairs")
    print(f"- Threshold-based filtered: {threshold_filtered_count} pairs")
    print(f"- Total filtered: {perfect_match_count + threshold_filtered_count} pairs")

    print("\n" + "=" * 60 + "\n")