"""
preprocess.py

Loads datasets from /data/raw/, normalizes them into a unified format:
[
  {"input_text": "...", "target_text": "..."},
  ...
]
and saves them into:
- /data/processed/core_corpus.json
- /data/processed/study_corpus.json

Supports nested MMLU datasets stored under:
data/raw/cais/mmlu_<subject>
"""
import os
import json
from datasets import load_from_disk

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CORE_FILE = os.path.join(PROCESSED_DIR, "core_corpus.json")
STUDY_FILE = os.path.join(PROCESSED_DIR, "study_corpus.json")

# -----------------------------
# Dataset-specific preprocessors
# -----------------------------
def preprocess_squad(dataset):
    samples = []
    for row in dataset.get("train", []):
        for ans in row.get("answers", {}).get("text", []):
            samples.append({
                "input_text": f"Question: {row.get('question', '')} Context: {row.get('context', '')}",
                "target_text": ans
            })
    return samples

def preprocess_race(dataset):
    samples = []
    for row in dataset.get("train", []):
        if "questions" in row and "options" in row and "answers" in row:
            for q, opts, ans in zip(row["questions"], row["options"], row["answers"]):
                samples.append({
                    "input_text": f"Passage: {row.get('article','')} Question: {q} Options: {', '.join(opts)}",
                    "target_text": ans
                })
        elif "question" in row and "options" in row and "answer" in row:
            samples.append({
                "input_text": f"Passage: {row.get('article','')} Question: {row['question']} Options: {', '.join(row['options'])}",
                "target_text": row["answer"]
            })
    return samples

def preprocess_arc(dataset):
    from datasets import Dataset, DatasetDict
    samples = []
    if isinstance(dataset, Dataset):
        data_splits = {"train": dataset}
    elif isinstance(dataset, DatasetDict):
        data_splits = dataset
    else:
        return []

    for split_name, split in data_splits.items():
        for row in split:
            if isinstance(row, dict):
                choices = [c["text"] if isinstance(c, dict) else str(c) for c in row.get("choices", [])]
                samples.append({
                    "input_text": f"Question: {row.get('question','')} Options: {', '.join(choices)}",
                    "target_text": row.get("answerKey","")
                })
    return samples

def preprocess_cnn_dailymail(dataset):
    return [{"input_text": row.get("article",""), "target_text": row.get("highlights","")} for row in dataset.get("train", [])]

def preprocess_scientific(dataset):
    if "train" in dataset and "article" in dataset["train"].column_names and "abstract" in dataset["train"].column_names:
        return [{"input_text": row["article"], "target_text": row["abstract"]} for row in dataset["train"]]
    return []

def preprocess_openbookqa(dataset):
    samples = []
    splits = dataset if isinstance(dataset, dict) else {"train": dataset}
    for split_name, split in splits.items():
        if not hasattr(split, "column_names"):
            continue
        if "question_stem" in split.column_names:
            for row in split:
                choices_str = ", ".join(f"{k}: {v}" for k, v in row["choices"].items()) if isinstance(row["choices"], dict) else str(row["choices"])
                samples.append({
                    "input_text": f"Question: {row['question_stem']} Choices: {choices_str}",
                    "target_text": row["answerKey"]
                })
    return samples

def preprocess_math(dataset):
    samples = []
    splits = dataset if isinstance(dataset, dict) else {"train": dataset}
    for split_name, split in splits.items():
        if not hasattr(split, "column_names"):
            continue
        input_col = "query" if "query" in split.column_names else "question" if "question" in split.column_names else None
        target_col = "answer" if "answer" in split.column_names else None
        if input_col and target_col:
            for row in split:
                samples.append({
                    "input_text": row[input_col],
                    "target_text": row[target_col]
                })
    return samples

def preprocess_svamp(dataset):
    samples = []
    splits = dataset if isinstance(dataset, dict) else {"train": dataset}
    for split_name, split in splits.items():
        if not hasattr(split, "column_names"):
            continue
        if "Question" in split.column_names and "Answer" in split.column_names:
            for row in split:
                q = row["Body"] + " " + row["Question"] if "Body" in split.column_names else row["Question"]
                samples.append({
                    "input_text": q,
                    "target_text": row["Answer"]
                })
    return samples

def preprocess_mmlu(dataset):
    samples = []
    for split in dataset.keys():
        for row in dataset[split]:
            samples.append({
                "input_text": f"Question: {row.get('question','')}",
                "target_text": row.get("answer","")
            })
    return samples

# -----------------------------
# Dataset groups
# -----------------------------
CORE_DATASETS = {
    "squad_v2": preprocess_squad,
    "race_high": preprocess_race,
    "race_middle": preprocess_race,
    "ai2_arc_ARC-Easy": preprocess_arc,
    "ai2_arc_ARC-Challenge": preprocess_arc,
    "cnn_dailymail_3.0.0": preprocess_cnn_dailymail,
    "openbookqa": preprocess_openbookqa,
}

STUDY_DATASETS = {
    "scifact": preprocess_scientific,
    "gsm8k_main": preprocess_math,
    "ChilleD_SVAMP": preprocess_svamp,
    "asdiv": preprocess_math,
    # All cais/mmlu_* will go here
}

# -----------------------------
# Processing helper
# -----------------------------
def process_dataset(full_path, dataset_name, preproc_map):
    from datasets import Dataset, DatasetDict
    try:
        dataset = load_from_disk(full_path)
    except Exception as e:
        print(f"  Failed to load {dataset_name}: {e}")
        return []
    if dataset_name in preproc_map:
        func = preproc_map[dataset_name]
    elif dataset_name.startswith("cais/mmlu_"):
        func = preprocess_mmlu
    else:
        print(f"  No preprocessor found for {dataset_name}, skipping.")
        return []
    try:
        return func(dataset)
    except Exception as e:
        print(f"  Preprocessing failed for {dataset_name}: {e}")
        return []

# -----------------------------
# Main
# -----------------------------
def preprocess_all():
    core_samples, study_samples = [], []

    for dataset_dir in sorted(os.listdir(RAW_DIR)):
        full_path = os.path.join(RAW_DIR, dataset_dir)
        if not os.path.isdir(full_path):
            continue

        if dataset_dir == "cais":
            for sub in sorted(os.listdir(full_path)):
                sub_path = os.path.join(full_path, sub)
                if os.path.isdir(sub_path) and sub.startswith("mmlu_"):
                    dataset_name = "cais/mmlu_" + sub[5:]
                    samples = process_dataset(sub_path, dataset_name, STUDY_DATASETS)
                    study_samples.extend(samples)
            continue

        if dataset_dir in CORE_DATASETS:
            samples = process_dataset(full_path, dataset_dir, CORE_DATASETS)
            core_samples.extend(samples)
        elif dataset_dir in STUDY_DATASETS:
            samples = process_dataset(full_path, dataset_dir, STUDY_DATASETS)
            study_samples.extend(samples)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CORE_FILE, "w", encoding="utf-8") as f:
        json.dump(core_samples, f, indent=2)
    with open(STUDY_FILE, "w", encoding="utf-8") as f:
        json.dump(study_samples, f, indent=2)

    print(f"\nFinal: {len(core_samples)} core samples → {CORE_FILE}")
    print(f"       {len(study_samples)} study samples → {STUDY_FILE}")

if __name__ == "__main__":
    preprocess_all()
