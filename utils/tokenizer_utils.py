"""
tokenizer_utils.py

Provides tokenization utilities for the StudyHelper-SLM project.
Automatically loads the trained custom tokenizer if available,
otherwise falls back to a base pretrained tokenizer (e.g., "gpt2").
Adds domain-specific special tokens for scientific/academic text.
"""

import os
from transformers import AutoTokenizer

# Global tokenizer instance (lazy-loaded)
_tokenizer = None

# Base pretrained tokenizer (fallback)
BASE_MODEL_NAME = "gpt2"

# Directory for custom trained tokenizer
CUSTOM_TOKENIZER_DIR = "models/tokenizer"

# Domain-specific tokens (extendable)
SPECIAL_TOKENS = ["<eq>", "<mol>", "<bio>", "<chem>"]


def get_tokenizer():
    """
    Loads and returns a Hugging Face AutoTokenizer.
    If a custom tokenizer exists, loads that; otherwise uses BASE_MODEL_NAME.

    Returns:
        transformers.PreTrainedTokenizer: Tokenizer instance with extended vocabulary and pad token.
    """
    global _tokenizer
    if _tokenizer is None:
        if os.path.exists(CUSTOM_TOKENIZER_DIR):
            print(f"📦 Loading custom tokenizer from {CUSTOM_TOKENIZER_DIR}")
            _tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_DIR)
        else:
            print(f"⚠️ Custom tokenizer not found, falling back to {BASE_MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Add domain-specific tokens (skip if already present)
        added = _tokenizer.add_tokens(SPECIAL_TOKENS)
        if added:
            print(f"Added {added} domain-specific special tokens: {SPECIAL_TOKENS}")

        # Ensure pad token exists (GPT-2 does not have one by default)
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            _tokenizer.pad_token = "[PAD]"
            print("pad_token was missing, added [PAD] token")

    return _tokenizer
