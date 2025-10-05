"""
train_tokenizer.py

Trains a custom Byte-Pair Encoding (BPE) tokenizer on the combined Core + Study corpora.
This ensures Core and StudyNet SLMs share the same aligned vocabulary.
"""

import os
import json
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# Input + output paths
CORE_FILE = "data/processed/core_corpus.json"
STUDY_FILE = "data/processed/study_corpus.json"
TOKENIZER_DIR = "models/tokenizer"


def train_tokenizer(
    vocab_size=52000,
    min_freq=2,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
):
    """
    Train a Byte-Pair Encoding (BPE) tokenizer on Core + Study datasets.

    Args:
        vocab_size (int): Target vocabulary size.
        min_freq (int): Minimum frequency for subwords to be included.
        special_tokens (list[str]): Special tokens to add to the tokenizer.
    """
    texts = []

    for file in [CORE_FILE, STUDY_FILE]:
        if not os.path.exists(file):
            print(f"⚠️ Skipping missing dataset: {file}")
            continue

        print(f"📂 Loading {file} ...")
        with open(file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for item in dataset:
            input_text = str(item.get("input_text", "")).strip()
            target_text = str(item.get("target_text", "")).strip()
            if input_text:
                texts.append(input_text)
            if target_text:
                texts.append(target_text)

    if not texts:
        raise ValueError("❌ No valid text found in datasets to train tokenizer.")

    print(f"⚡ Training BPE tokenizer on {len(texts)} sequences...")
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        iterator=texts,
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens
    )

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)  # vocab.json + merges.txt
    tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(TOKENIZER_DIR, "tokenizer.json"),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    hf_tokenizer.save_pretrained(TOKENIZER_DIR)

    print(f"✅ Tokenizer trained and saved to {TOKENIZER_DIR}")


if __name__ == "__main__":
    train_tokenizer()
