"""
tokenize.py

Converts processed datasets (Core + Study) into tokenized format ready for training.
Concatenates input + target with proper masking so that loss is only computed on target tokens.
Uses the tokenizer from utils/tokenizer_utils.py so special tokens are consistently applied.
"""

import json
import os
from utils.tokenizer_utils import get_tokenizer

# Input / output file paths + configs
DATASETS = {
    "core": {
        "input": "data/processed/core_corpus.json",
        "output": "data/tokenized/core_data_tokenized.json",
        "max_input_len": 512,
        "max_target_len": 128,
    },
    "study": {
        "input": "data/processed/study_corpus.json",
        "output": "data/tokenized/study_data_tokenized.json",
        "max_input_len": 1024,
        "max_target_len": 256,
    },
}


def tokenize_file(input_file, output_file, tokenizer, max_input_len=512, max_target_len=128):
    """Helper to tokenize a single dataset JSON file with input-target concatenation and masking."""
    if not os.path.exists(input_file):
        print(f"⚠️ Skipping missing dataset: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenized = []
    for item in dataset:
        input_text = str(item.get("input_text", "")).strip()
        target_text = str(item.get("target_text", "")).strip()

        if not input_text or not target_text:
            continue  # skip empty entries

        # Tokenize input
        input_enc = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=max_input_len,
        )

        # Tokenize target
        target_enc = tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=max_target_len,
        )

        # Concatenate input + target
        input_ids = input_enc["input_ids"] + target_enc["input_ids"]
        attention_mask = input_enc["attention_mask"] + target_enc["attention_mask"]

        # Labels: mask input portion with -100 for loss calculation
        labels = [-100] * len(input_enc["input_ids"]) + [
            (tok if tok != tokenizer.pad_token_id else -100) for tok in target_enc["input_ids"]
        ]

        # Truncate to max allowed length if necessary
        max_total_len = max_input_len + max_target_len
        input_ids = input_ids[:max_total_len]
        attention_mask = attention_mask[:max_total_len]
        labels = labels[:max_total_len]

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokenized, f, indent=2)

    print(f"✅ Tokenized dataset saved to {output_file} ({len(tokenized)} samples).")


def tokenize_datasets():
    """Tokenize both Core + Study datasets with the shared tokenizer."""
    tokenizer = get_tokenizer()

    # Ensure tokenizer has required special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token = "<pad>"
        print("pad_token was missing, added <pad>")
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})
        tokenizer.eos_token = "<eos>"
        print("eos_token was missing, added <eos>")

    for name, paths in DATASETS.items():
        print(f"\n📦 Processing {name} dataset...")
        tokenize_file(
            paths["input"],
            paths["output"],
            tokenizer,
            max_input_len=paths["max_input_len"],
            max_target_len=paths["max_target_len"],
        )

    print(f"\n📚 Vocab size: {len(tokenizer)} (with special tokens)")


if __name__ == "__main__":
    tokenize_datasets()
