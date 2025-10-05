"""
Fetches open academic and QA datasets from HuggingFace (SQuAD, RACE, ARC, etc.)
and saves them into /data/raw/ for later preprocessing.

The script attempts to download multiple datasets and store them locally using
the Hugging Face `datasets` library. Datasets that are unavailable or fail to
download are skipped, with errors logged.
"""

from datasets import load_dataset
import os

# Directory to save raw datasets
RAW_DIR = "data/raw"

# List of datasets to download
DATASETS = {
    "squad_v2": {"name": "squad_v2", "subset": None, "type": "QA"},
    "race_high": {"name": "race", "subset": "high", "type": "QA"},
    "race_middle": {"name": "race", "subset": "middle", "type": "QA"},
    "arc_easy": {"name": "ai2_arc", "subset": "ARC-Easy", "type": "science"},
    "arc_challenge": {"name": "ai2_arc", "subset": "ARC-Challenge", "type": "science"},
    "cnn_dailymail": {"name": "cnn_dailymail", "subset": "3.0.0", "type": "summarization"},
    "openbookqa": {"name": "openbookqa", "subset": None, "type": "science"},
    "scifact": {"name": "scifact", "subset": None, "type": "science"},
    "gsm8k": {"name": "gsm8k", "subset": "main", "type": "math"},
    "svamp": {"name": "ChilleD/SVAMP", "subset": None, "type": "math"},
    "mmlu": {"name": "cais/mmlu", "subset": None, "type": "multi-subject"},  # Will handle subjects dynamically
}

# All available MMLU subject configs
MMLU_SUBJECTS = [
    'abstract_algebra', 'all', 'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
    'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies', 'sociology',
    'us_foreign_policy', 'virology', 'world_religions'
]

def download_and_save(dataset_name: str, subset: str = None, save_dir: str = RAW_DIR) -> bool:
    """
    Download a dataset from Hugging Face and save it locally.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        subset (str, optional): Optional subset or configuration of the dataset.
        save_dir (str): Directory where the dataset will be saved.

    Returns:
        bool: True if download and save were successful, False otherwise.
    """
    try:
        print(f"Downloading {dataset_name} ({'subset: ' + subset if subset else 'full'})...")
        dataset = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        target_dir = os.path.join(save_dir, dataset_name + (f"_{subset}" if subset else ""))
        os.makedirs(target_dir, exist_ok=True)
        dataset.save_to_disk(target_dir)
        print(f"Saved {dataset_name} → {target_dir}")
        return True
    except Exception as e:
        print(f"Failed to download {dataset_name} ({subset}): {type(e).__name__} - {e}")
        return False

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    success_count = 0

    for key, cfg in DATASETS.items():
        if cfg["name"] == "cais/mmlu" and cfg["subset"] is None:
            # Loop over all subjects for MMLU
            for subject in MMLU_SUBJECTS:
                if download_and_save(cfg["name"], subject):
                    success_count += 1
        else:
            if download_and_save(cfg["name"], cfg["subset"]):
                success_count += 1

    print(f"\nFinished downloading. {success_count}/{len(DATASETS) + len(MMLU_SUBJECTS)-1} datasets saved.")
