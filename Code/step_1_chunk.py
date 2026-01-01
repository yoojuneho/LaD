# chunk.py
# Step-1
# Generate all possible sentence pairs between [news1, conflict_news, other_news1, other_news2, other_news3]
# Sentence pairs within the same document are not generated

import json
import os
import spacy
from pathlib import Path

# ============================================================
# PATH SETTINGS
# ============================================================

# Using Path for better cross-platform compatibility
BASE_DIR = Path("input Low Intensity Factoid Conflict Dataset Folder Path")
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "Low_Intensity_Factoid_Conflict.json"
OUTPUT_DIR = DATA_DIR / "Chunk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD spaCy
# ============================================================
NLP = spacy.load("en_core_web_sm")


# ============================================================
# 1. Sentence Chunking
# ============================================================
def chunk_sentences(text, doc_id):
    doc = NLP(text)
    results = []
    for sent in doc.sents:
        t = sent.text.strip()
        if t:
            results.append({"text": t, "doc_id": doc_id})
    return results


# ============================================================
# 2. Generate ALL Document Sentence Pairs
# ============================================================
def generate_all_sentence_pairs(doc_sentences):
    """
    doc_sentences: dict
        {
            "news1": [sent1, sent2, ...],
            "conflict_news": [sent1, sent2, ...],
            ...
        }
    """

    doc_keys = list(doc_sentences.keys())
    pairs = []
    pair_id = 1

    # All document combinations (without A-B, B-A duplicates)
    for i in range(len(doc_keys)):
        for j in range(i + 1, len(doc_keys)):
            docA = doc_keys[i]
            docB = doc_keys[j]

            sentsA = doc_sentences[docA]
            sentsB = doc_sentences[docB]

            # Cartesian product of all sentences
            for s1 in sentsA:
                for s2 in sentsB:
                    pairs.append({
                        "id": pair_id,
                        "sentence_1": s1,
                        "sentence_2": s2,
                        "semantic_score": "",
                        "semantic_filtered": "",
                        "cosine_score": "",
                        "cosine_filtered": "",
                        "conflict_label": ""
                    })
                    pair_id += 1

    return pairs


# ============================================================
# 3. Process Single Event
# ============================================================
def process_event(event):
    event_id = event["id"]
    print(f"=== Processing event {event_id} ===")

    news = event["news"]

    # document keys to include in ALL-to-ALL matching
    target_keys = [
        "news1",
        "conflict_news",
        "other_news1",
        "other_news2",
        "other_news3",
    ]

    # 1) Sentence chunking for all target documents
    doc_sentences = {}
    for key in target_keys:
        if key in news and "article" in news[key]:
            doc_sentences[key] = chunk_sentences(news[key]["article"], key)

    # 2) Generate ALL-to-ALL document sentence pairs
    pairs = generate_all_sentence_pairs(doc_sentences)

    print(f"Generated {len(pairs)} sentence pairs (ALL combinations)")

    # Save output
    output_file = os.path.join(OUTPUT_DIR, f"event_{event_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_file}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading dataset:", INPUT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for event in data:
        process_event(event)
