"""
dataset.py

PyTorch Dataset wrapper for loading tokenized Core/Study data
into batches for training the SLM models.
"""

import json
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    def __init__(self, tokenized_file: str):
        """
        Args:
            tokenized_file (str): Path to tokenized JSON/NDJSON file (core or study).
        """
        self.data = []

        try:
            if tokenized_file.endswith(".ndjson"):
                # Read line by line (each line is a JSON object)
                with open(tokenized_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line))
            else:
                # Default: JSON array
                with open(tokenized_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"[ERR] Tokenized file not found: {tokenized_file}")
        except json.JSONDecodeError:
            raise ValueError(f"[ERR] Failed to parse tokenized file: {tokenized_file}")

        if not isinstance(self.data, list):
            raise ValueError(f"[ERR] Tokenized dataset format invalid in: {tokenized_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }
