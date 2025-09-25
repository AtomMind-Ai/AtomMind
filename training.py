"""
AtomMind Training Script (Fully Fixed & Optimized):
- Load dataset from datasets/dataset.txt
- Train SmallScientificLLM with MultiNetMemory
- Gradient accumulation, token dropout, AMP, LR scheduler, early stopping
- Optional validation split
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.nn import Embedding
from torch.utils.data import DataLoader, random_split

from memory.memory import MemorySystem
from models.slm import SmallLanguageModel
from tokenizer import tokenizer
from utils.logger import Logger
from chatconfig import DEVICE, HIDDEN_SIZE, SAVE_PATH, DOMAINS
from data.loader import TextDataset, collate_fn

# -------------------- Config --------------------
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0
EPOCHS = 100
PATIENCE = 3
ACCUM_STEPS = 8
TOKEN_DROPOUT = 0.15
DATA_FILE = "datasets/dataset.txt"
VALIDATION_SPLIT = 0.1

os.makedirs("datasets", exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------- Check dataset --------------------
if not os.path.isfile(DATA_FILE):
    print(f"[Error] No dataset found at '{DATA_FILE}'. Exiting.")
    sys.exit(1)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    dataset_lines = [line.strip() for line in f if line.strip()]

if len(dataset_lines) == 0:
    print(f"[Error] Dataset file '{DATA_FILE}' is empty. Exiting.")
    sys.exit(1)

# -------------------- Ensure tokenizer pad token --------------------
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0

# -------------------- Initialize --------------------
logger = Logger()
slm = SmallLanguageModel().to(DEVICE)
memory = MemorySystem(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    device=DEVICE,
)
embedding_layer = Embedding(tokenizer.vocab_size, HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(slm.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(enabled=use_amp)

# -------------------- Dataset / DataLoader --------------------
dataset = TextDataset(dataset_lines)

# Validation split
val_size = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) if val_size > 0 else (dataset, None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) if val_dataset else None

# -------------------- Training Loop --------------------
best_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    logger.log(f"\n=== Epoch {epoch+1} ===")
    total_loss = 0.0
    slm.train()
    optimizer.zero_grad()

    for i, tokens in enumerate(train_loader):
        tokens = tokens.long().to(DEVICE)

        # ---- Token Dropout ----
        if TOKEN_DROPOUT > 0:
            mask = torch.rand(tokens.shape, device=DEVICE) < TOKEN_DROPOUT
            tokens_input = tokens.clone()
            tokens_input[mask] = tokenizer.pad_token_id
        else:
            tokens_input = tokens

        input_tensor = embedding_layer(tokens_input)
        x_dict = {domain: input_tensor for domain in DOMAINS}

        encoded = memory.store_batch(tokens)
        recalled_vec = memory.associative.retrieve(encoded).unsqueeze(1)

        # ---- AMP Forward ----
        with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
            output = slm(x_dict, chat_tensor=recalled_vec)
            logits = torch.matmul(output, embedding_layer.weight.T)
            seq_len = min(logits.size(1), tokens.size(1)) - 1
            if seq_len <= 0:
                logger.log(f"[Warning] Skipping batch (short sequence: {tokens.size(1)})")
                continue

            logits = logits[:, :seq_len, :].contiguous()
            targets = tokens[:, 1: seq_len + 1].contiguous()
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1), ignore_index=0)
            loss = loss / ACCUM_STEPS  # normalize for gradient accumulation

        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS

        # ---- Gradient Accumulation ----
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(slm.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if i % 10 == 0:
            logger.log(f"[Epoch {epoch+1} | Batch {i+1}/{len(train_loader)}] Loss: {loss.item() * ACCUM_STEPS:.4f}")

    # ---- Handle last incomplete accumulation ----
    if len(train_loader) % ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(slm.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # ---- Validation ----
    avg_val_loss = None
    if val_loader:
        slm.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for tokens in val_loader:
                tokens = tokens.long().to(DEVICE)
                input_tensor = embedding_layer(tokens)
                x_dict = {domain: input_tensor for domain in DOMAINS}
                encoded = memory.store_batch(tokens)
                recalled_vec = memory.associative.retrieve(encoded).unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = slm(x_dict, chat_tensor=recalled_vec)
                    logits = torch.matmul(output, embedding_layer.weight.T)
                    seq_len = min(logits.size(1), tokens.size(1)) - 1
                    if seq_len <= 0:
                        continue
                    logits = logits[:, :seq_len, :].contiguous()
                    targets = tokens[:, 1: seq_len + 1].contiguous()
                    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1), ignore_index=0)
                    val_loss_total += loss.item() * tokens.size(0)
        avg_val_loss = val_loss_total / len(val_dataset)
        logger.log(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

    avg_loss = total_loss / max(1, len(train_loader))
    logger.log(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    logger.log(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ---- Scheduler Step ----
    scheduler.step(avg_val_loss if avg_val_loss is not None else avg_loss)

    # ---- Early Stopping ----
    monitor_loss = avg_val_loss if avg_val_loss is not None else avg_loss
    if monitor_loss < best_loss:
        best_loss = monitor_loss
        patience_counter = 0
        torch.save(slm.state_dict(), os.path.join(SAVE_PATH, "slm_pretrained.pt"))
        torch.save(memory.state_dict(), os.path.join(SAVE_PATH, "memory_pretrained.pt"))
        logger.log("Model improved & saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.log("[Early Stop] Validation stalled")
            break

# ---- Final Save ----
torch.save(slm.state_dict(), os.path.join(SAVE_PATH, "slm_final.pt"))
torch.save(memory.state_dict(), os.path.join(SAVE_PATH, "memory_final.pt"))
logger.log("Training completed!")
