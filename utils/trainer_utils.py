# utils/trainer_utils.py

import torch
import numpy as np
from transformers import set_seed

def set_global_seed(seed=42):
    """Ensure reproducibility across runs."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_base_model(model):
    """Freeze all parameters except adapters (if present)."""
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

def save_checkpoint(model, tokenizer, save_dir):
    """Save model and tokenizer."""
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Saved checkpoint to {save_dir}")
