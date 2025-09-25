"""
Data generator + loader for AtomMind training.
Handles synthetic dataset creation with ChatGenerator + ExecutorAgent,
and prepares PyTorch Dataset + DataLoader utilities.
"""

import os
import torch
from torch.utils.data import Dataset
from agents.generators import ChatGenerator
from agents.executor import ExecutorAgent
from tokenizer import tokenizer
from utils.tokens import adjust_seq_len
from utils.logger import Logger
from chatconfig import MAX_SEQ_LEN, DEVICE

logger = Logger()

DATASET_PATH = "datasets/dataset.txt"
MAX_TOKENS_PER_CALL = 1500


# -------------------- Dataset Class -------------------- #
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sample = self.texts[idx]
        tokens = tokenizer.encode(sample, return_tensors="pt").squeeze(0)
        return tokens


def collate_fn(batch):
    tokens = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    tokens = adjust_seq_len(tokens, MAX_SEQ_LEN)
    return tokens.to(DEVICE)


# -------------------- Dataset Generator -------------------- #
def generate_dataset(num_samples=50):
    """
    Generate synthetic dataset and save to file.
    - Appends to existing file if present.
    - Creates a new file if none exists.

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        list[str]: Generated dataset lines
    """
    executor = ExecutorAgent()
    chat_gen = ChatGenerator(executor)
    dataset = []

    for _ in range(num_samples):
        text = chat_gen.generate(num_candidates=1)[0]
        tokens = tokenizer.encode(text)[:MAX_TOKENS_PER_CALL]
        dataset.append(tokenizer.decode(tokens))

    os.makedirs("datasets", exist_ok=True)
    mode = "a"  # Always append
    with open(DATASET_PATH, mode, encoding="utf-8") as f:
        for item in dataset:
            f.write(item + "\n")

    logger.log(f"Dataset updated: {len(dataset)} new samples -> {DATASET_PATH}")
    return dataset


# -------------------- Loader -------------------- #
def load_dataset():
    """
    Load dataset from file.

    Returns:
        list[str]: Dataset lines
    """
    if not os.path.exists(DATASET_PATH):
        logger.log(f"[Info] Dataset file '{DATASET_PATH}' not found. Returning empty list.")
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [line.strip() for line in f if line.strip()]

    logger.log(f"Loaded dataset: {len(dataset)} samples")
    return dataset
