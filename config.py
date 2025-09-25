"""
Configuration file for AtomMind project (daily chat mode).
NOTE: Good for fast and lightweight training with SLM.
IMPORTANT/WARNING: DONT USE AT THE SAME TIME WITH `config.py`.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
HIDDEN_SIZE = 256
NUM_ATTENTION_HEADS = 8

# Layer counts (reduced for efficiency in daily chat)
DEN_LAYERS = 6   # Domain Expert Network (shrunk)
GKB_LAYERS = 6   # General Knowledge Backbone
SRM_LAYERS = 4   # Symbolic Reasoning Module
OAM_LAYERS = 4   # Optimization & Algorithmic Module
CEN_LAYERS = 6   # Chat Expert Network

# Sequence length
MAX_SEQ_LEN = 256

# Training hyperparameters
BATCH_SIZE = 8         # larger batch for stability
LEARNING_RATE = 3e-4   # slightly faster convergence
EPOCHS = 5

# Supported domains (collapsed to single domain for chat)
DOMAINS = ["chat"]

# Paths
DATA_PATH = "./data/"
LOG_PATH = "./logs/"
SAVE_PATH = "./checkpoints/"
