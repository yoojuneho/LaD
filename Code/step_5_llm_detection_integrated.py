# llm_detection_integrated.py
# Execute Baseline and Guided approaches simultaneously and output comparison results

import os, json, random, time, re
from tqdm import tqdm
from openai import OpenAI

# ===========================
# Basic Configuration
# ===========================
# Path settings - modify these according to your environment
DATA_PATH = "./data/data.json"  # Main dataset file
CHUNK_DIR = "./data/sentence_detected"  # Input from sentence detection output
SAVE_DIR = "./data/document_detected"  # Output directory for document detection
os.makedirs(SAVE_DIR, exist_ok=True)

# API key should be set via environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

MODELS = ["gpt-3.5-turbo"]
GEN_PARAMS = dict(temperature=0.0, max_tokens=512)
DOC_COUNT = 5

# ===========================
# Controlled Random Seeds
# ===========================
MASTER_SEED = 42
SAMPLE_ORDER_SEED = 100
ID_SHUFFLE_SEED_BASE = 1000


# ================================================================
# Load Candidate Sentence Pairs
# ================================================================
def load_candidate_pairs(event_id):
    path = os.path.join(CHUNK_DIR, f"event_{event_id}.json")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    cands = []
    for p in pairs:
        if p.get("conflict_label") == 1:
            cands.append({
                "s1": p["sentence_1"]["text"],
                "s2": p["sentence_2"]["text"],
                "doc_id_1": p["sentence_1"]["doc_id"],
                "doc_id_2": p["sentence_2"]["doc_id"],
            })
    return cands


# ================================================================
# Document Formatting
# ================================================================
def format_docs_baseline(docs):
    return "\n\n".join(
        f"[Doc {i}] ({tag}): {txt}"
        for i, (tag, txt, _) in enumerate(docs, 1)
    )


def format_docs_guided(docs):
    return "\n\n".join(
        f"[Doc {i}] ({tag}, id={doc_id}): {txt}"
        for i, (tag, txt, doc_id) in enumerate(docs, 1)
    )


def map_candidates_to_indices(cands, doc_id_to_idx):
    used, dropped = [], []
    for c in cands:
        i1 = doc_id_to_idx.get(c["doc_id_1"])
        i2 = doc_id_to_idx.get(c["doc_id_2"])
        if not i1 or not i2:
            dropped.append(c)
            continue
        used.append({
            "s1": c["s1"], "s2": c["s2"],
            "i1": i1, "i2": i2
        })
    return used, dropped


def format_focus_pairs(mapped):
    if not mapped:
        return ""
    out = []
    for i, c in enumerate(mapped, 1):
        out.append(
            f"[Pair {i}]\n"
            f"Sentence A (Doc {c['i1']}): {c['s1']}\n"
            f"Sentence B (Doc {c['i2']}): {c['s2']}"
        )
    return "\n\n".join(out)


# ================================================================
# Prompts
# ================================================================
PROMPT_BASE = """
You are an expert factual conflict detector.

You will be given 5 documents (Doc 1..5). Some may describe the SAME event; others may be unrelated.

Your task:
1) Determine whether any documents contain factual contradictions about the SAME event or entity.
2) Translation/paraphrase differences are NOT contradictions.
3) Only explicit factual disagreements count.

Analyze the following documents:
{docs}

Output ONLY valid JSON:
{{
  "conflict_exists": true/false,
  "conflicting_docs": [integers],
  "reason": "one concise sentence"
}}
""".strip()


PROMPT_WITH_HINTS = """
You are an expert factual conflict detector.

You will be given 5 documents (Doc 1..5). Some may describe the SAME event; others may be unrelated.

Your task:
1) Determine whether any documents contain factual contradictions about the SAME event or entity.
2) Translation/paraphrase differences are NOT contradictions.
3) Only explicit factual disagreements count.

Analyze the following documents:
{docs}

IMPORTANT: Some sentence pairs are provided below for reference only. These may or may not indicate actual conflicts. 
You MUST independently analyze ALL documents and make your own judgment.

{focus}

Output ONLY valid JSON:
{{
  "conflict_exists": true/false,
  "conflicting_docs": [integers],
  "reason": "one concise sentence"
}}
""".strip()


# ================================================================
# Utilities
# ================================================================
def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.DOTALL).strip()
    f, l = text.find("{"), text.rfind("}")
    if f != -1 and l != -1:
        try: return json.loads(text[f:l+1])
        except: return {}
    return {}

def to_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() in {"true", "1", "yes"}
    return False

def intify_list(xs):
    if xs is None: return []
    if isinstance(xs, (int,str)): xs = [xs]
    out=[]
    for v in xs:
        if isinstance(v,int): k=v
        else:
            m=re.search(r"\d+", str(v))
            if not m: continue
            k=int(m.group())
        if 1<=k<=DOC_COUNT: out.append(k)
    return sorted(set(out))

def call_gpt(model, prompt):
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system", "content":"Return ONLY valid JSON."},
                    {"role":"user", "content":prompt}
                ],
                **GEN_PARAMS
            )
            return resp.choices[0].message.content
        except Exception as e:
            print("Error:", e)
            time.sleep(3)
    return "{}"


# ================================================================
# Generate Gold Samples
# ================================================================
def build_conflict_sample(item):
    n = item["news"]
    docs=[
        ("news1", n["news1"]["article"], "news1"),
        ("news2_conflict", n["conflict_news"]["article"], "conflict_news"),
        ("other1", n["other_news1"]["article"], "other_news1"),
        ("other2", n["other_news2"]["article"], "other_news2"),
        ("other3", n["other_news3"]["article"], "other_news3"),
    ]
    
    # Fixed seed per ID
    seed = ID_SHUFFLE_SEED_BASE + hash(item["id"]) % 10000
    random.seed(seed)
    random.shuffle(docs)

    gold_exists = True
    gold_docs = []
    for i, (tag, _, _) in enumerate(docs, 1):
        if tag in ["news1", "news2_conflict"]:
            gold_docs.append(i)
    
    return docs, gold_exists, sorted(gold_docs)


def build_non_conflict_sample(item):
    n=item["news"]
    docs=[
        ("news1", n["news1"]["article"], "news1"),
        ("news2_para", n["news2"]["article"], "news2_para"),
        ("other1", n["other_news1"]["article"], "other_news1"),
        ("other2", n["other_news2"]["article"], "other_news2"),
        ("other3", n["other_news3"]["article"], "other_news3"),
    ]
    
    # Fixed seed per ID
    seed = ID_SHUFFLE_SEED_BASE + hash(item["id"]) % 10000
    random.seed(seed)
    random.shuffle(docs)
    
    return docs, False, []


# ================================================================
# Strict Evaluation
# ================================================================
def evaluate_strict(pred_exists, pred_docs, gold_exists, gold_docs):
    if pred_exists != gold_exists: return 0
    if not gold_exists: return int(pred_docs == [])
    return int(sorted(pred_docs) == sorted(gold_docs))


# ================================================================
# Run Single Experiment
# ================================================================
def run_experiment(model, expanded, experiment_mode):
    """
    experiment_mode: "baseline" or "guided"
    """
    print(f"\n{'='*60}")
    print(f"Running {experiment_mode.upper()} experiment")
    print(f"{'='*60}")
    
    results = []
    total_conflict = correct_conflict = 0
    total_nonconf = correct_nonconf = 0

    for item, mode in tqdm(expanded, desc=f"{experiment_mode}-{model}"):
        
        if mode == "conflict":
            docs, gold_exists, gold_docs = build_conflict_sample(item)
        else:
            docs, gold_exists, gold_docs = build_non_conflict_sample(item)

        # ====================================
        # Generate Prompt
        # ====================================
        if experiment_mode == "baseline":
            docs_text = format_docs_baseline(docs)
            prompt = PROMPT_BASE.format(docs=docs_text)
            used_pairs = 0
            
        else:  # guided
            docs_text = format_docs_guided(docs)
            doc_id_map = {doc_id:i for i,(tag,_,doc_id) in enumerate(docs,1)}
            
            if mode == "conflict":
                raw_pairs = load_candidate_pairs(item["id"])
                mapped_pairs, _ = map_candidates_to_indices(raw_pairs, doc_id_map)
            else:
                mapped_pairs = []
            
            if len(mapped_pairs) >= 1:
                focus_text = format_focus_pairs(mapped_pairs)
                prompt = PROMPT_WITH_HINTS.format(
                    docs=docs_text,
                    focus=focus_text
                )
            else:
                prompt = PROMPT_BASE.format(docs=docs_text)
            
            used_pairs = len(mapped_pairs)

        # ====================================
        # Call LLM
        # ====================================
        raw_out = call_gpt(model, prompt)
        parsed = parse_json(raw_out)

        pred_exists = to_bool(parsed.get("conflict_exists"))
        pred_docs = intify_list(parsed.get("conflicting_docs", []))

        full_correct = evaluate_strict(pred_exists, pred_docs, gold_exists, gold_docs)

        if mode == "conflict":
            total_conflict += 1
            correct_conflict += full_correct
        else:
            total_nonconf += 1
            correct_nonconf += full_correct

        results.append({
            "id": item["id"],
            "mode": mode,
            "model": model,
            "experiment_mode": experiment_mode,
            "perm_order": [tag for tag,_,_ in docs],
            "gold_exists": gold_exists,
            "gold_docs": gold_docs,
            "pred_exists": pred_exists,
            "pred_docs": pred_docs,
            "full_correct": full_correct,
            "used_pairs": used_pairs,
            "raw_output": raw_out
        })

    overall_acc = sum(r["full_correct"] for r in results) / len(results)
    conflict_acc = correct_conflict / total_conflict if total_conflict > 0 else 0
    nonconf_acc = correct_nonconf / total_nonconf if total_nonconf > 0 else 0

    return {
        "overall": overall_acc,
        "conflict": conflict_acc,
        "nonconflict": nonconf_acc,
        "results": results
    }


# ================================================================
# MAIN
# ================================================================
def main():

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict): 
        data = [data]

    # Generate fixed sample order
    random.seed(SAMPLE_ORDER_SEED)
    expanded = []
    for item in data:
        expanded.append((item, "conflict"))
        expanded.append((item, "nonconflict"))
    random.shuffle(expanded)

    for model in MODELS:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model}")
        print(f"{'#'*70}")

        # Baseline Experiment
        baseline_results = run_experiment(model, expanded, "baseline")
        
        # Guided Experiment
        guided_results = run_experiment(model, expanded, "guided")

        # ====================================
        # Save Results
        # ====================================
        baseline_dir = os.path.join(SAVE_DIR, "baseline")
        guided_dir = os.path.join(SAVE_DIR, "guided")
        os.makedirs(baseline_dir, exist_ok=True)
        os.makedirs(guided_dir, exist_ok=True)

        # Save Baseline
        baseline_path = os.path.join(
            baseline_dir, 
            f"llm_detection_{model.replace('-', '_')}.json"
        )
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        
        # Save Guided
        guided_path = os.path.join(
            guided_dir,
            f"llm_detection_{model.replace('-', '_')}.json"
        )
        with open(guided_path, "w", encoding="utf-8") as f:
            json.dump(guided_results, f, indent=2, ensure_ascii=False)

        # ====================================
        # Output Comparison Results
        # ====================================
        print(f"\n{'='*70}")
        print(f"COMPARISON RESULTS - {model}")
        print(f"{'='*70}\n")

        print(f"{'Metric':<25} {'Baseline':<15} {'Guided':<15} {'Improvement':<15}")
        print(f"{'-'*70}")
        
        baseline_overall = baseline_results["overall"]
        guided_overall = guided_results["overall"]
        improve_overall = guided_overall - baseline_overall
        print(f"{'Overall Accuracy':<25} {baseline_overall:<15.3f} {guided_overall:<15.3f} {improve_overall:+.3f}")
        
        baseline_conflict = baseline_results["conflict"]
        guided_conflict = guided_results["conflict"]
        improve_conflict = guided_conflict - baseline_conflict
        print(f"{'Conflict Accuracy':<25} {baseline_conflict:<15.3f} {guided_conflict:<15.3f} {improve_conflict:+.3f}")
        
        baseline_nonconf = baseline_results["nonconflict"]
        guided_nonconf = guided_results["nonconflict"]
        improve_nonconf = guided_nonconf - baseline_nonconf
        print(f"{'NonConflict Accuracy':<25} {baseline_nonconf:<15.3f} {guided_nonconf:<15.3f} {improve_nonconf:+.3f}")
        
        print(f"\n{'='*70}\n")

        # ====================================
        # Save Comparison Results
        # ====================================
        comparison_dir = os.path.join(SAVE_DIR, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        comparison_data = {
            "model": model,
            "baseline": {
                "overall": baseline_overall,
                "conflict": baseline_conflict,
                "nonconflict": baseline_nonconf
            },
            "guided": {
                "overall": guided_overall,
                "conflict": guided_conflict,
                "nonconflict": guided_nonconf
            },
            "improvement": {
                "overall": improve_overall,
                "conflict": improve_conflict,
                "nonconflict": improve_nonconf
            }
        }
        
        comparison_path = os.path.join(
            comparison_dir,
            f"comparison_{model.replace('-', '_')}.json"
        )
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Baseline saved → {baseline_path}")
        print(f"✅ Guided saved → {guided_path}")
        print(f"✅ Comparison saved → {comparison_path}\n")


if __name__ == "__main__":
    main()