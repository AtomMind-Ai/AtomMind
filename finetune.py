"""
Fine-tuning script:
- Load pretrained SmallScientificLLM + MultiNetMemory
- Full OpenRouter control & logging
- Controller supports batch-level control & module freezing/unfreezing
"""

import os
import torch
from torch.nn import Embedding
import torch.nn.functional as F

from memory.memory import MemorySystem
from models.slm import SmallScientificLLM
from tokenizer import tokenizer
from utils.tokens import adjust_seq_len
from utils.logger import Logger
from chatconfig import DEVICE, MAX_SEQ_LEN, HIDDEN_SIZE, SAVE_PATH, DOMAINS
from agents.controller import ControllerAgent
from agents.generators import ChatGenerator
from agents.executor import ExecutorAgent

# -------------------- Initialize --------------------
logger = Logger()
slm = SmallScientificLLM().to(DEVICE)
memory = MemorySystem(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    device=DEVICE
)

# Load pretrained weights
slm.load_state_dict(torch.load(os.path.join(SAVE_PATH, "slm_pretrained.pt"), map_location=DEVICE))
memory.load_state_dict(torch.load(os.path.join(SAVE_PATH, "memory_pretrained.pt"), map_location=DEVICE))
logger.log("Loaded pretrained model and memory")

executor = ExecutorAgent()
chat_gen = ChatGenerator(executor)
controller = ControllerAgent(executor)
embedding_layer = Embedding(tokenizer.vocab_size, HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(slm.parameters(), lr=1e-5)  # smaller LR for fine-tuning

MAX_GRAD_NORM = 1.0
FINE_TUNE_EPOCHS = 3

# -------------------- Fine-tuning Loop --------------------
for epoch in range(FINE_TUNE_EPOCHS):
    logger.log(f"\n=== Fine-tune Epoch {epoch+1} ===")
    plan = controller.control_stage(
        stage="epoch_start",
        context={"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"], "grad_clip": MAX_GRAD_NORM},
    )

    # Controller adjustments (epoch-level)
    if "lr" in plan:
        for g in optimizer.param_groups:
            g["lr"] = plan["lr"]
        logger.log(f"[Controller] Adjusted learning rate -> {plan['lr']}")
    if "grad_clip" in plan:
        MAX_GRAD_NORM = plan["grad_clip"]
        logger.log(f"[Controller] Adjusted gradient clip -> {MAX_GRAD_NORM}")

    # Initialize accumulators for average epoch loss
    epoch_loss_total = 0.0
    num_batches = 0

    # Generate fine-tuning samples
    chat_samples = chat_gen.generate(num_candidates=1)
    for sample in chat_samples:
        tokens = tokenizer.encode(sample, return_tensors="pt").to(DEVICE).long()
        tokens = adjust_seq_len(tokens, MAX_SEQ_LEN)

        input_tensor = embedding_layer(tokens)  # [B, T, H]
        x_dict = {domain: input_tensor for domain in DOMAINS}

        # ---- Memory ----
        encoded = memory.encoder(tokens)              # [B, H]
        recalled_vec = memory.associative.retrieve(encoded)  # [B, H]
        recalled_vec = recalled_vec.unsqueeze(1)      # [B, 1, H]

        # ---- Forward ----
        output = slm(x_dict, chat_tensor=recalled_vec)  # [B, L, H]

        # ---- Loss (align lengths) ----
        seq_len = min(output.size(1), input_tensor.size(1))
        loss = F.mse_loss(output[:, :seq_len, :], input_tensor[:, :seq_len, :])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(slm.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        memory.store_batch(encoded.detach())
        logger.log(f"[Fine-tune] Loss: {loss.item():.4f}")

        # Accumulate loss for epoch average
        epoch_loss_total += loss.item()
        num_batches += 1

        # -------------------- Batch-level Controller --------------------
        batch_feedback = controller.control_stage(
            stage="batch_end",
            context={
                "epoch": epoch+1,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_clip": MAX_GRAD_NORM,
            },
        )

        # Apply controller feedback
        if "lr" in batch_feedback:
            for g in optimizer.param_groups:
                g["lr"] = batch_feedback["lr"]
            logger.log(f"[Controller] (Batch) Adjusted LR -> {batch_feedback['lr']}")

        if "grad_clip" in batch_feedback:
            MAX_GRAD_NORM = batch_feedback["grad_clip"]
            logger.log(f"[Controller] (Batch) Adjusted Grad Clip -> {MAX_GRAD_NORM}")

        if "freeze_modules" in batch_feedback:
            for module_name in batch_feedback["freeze_modules"]:
                if hasattr(slm, module_name):
                    for p in getattr(slm, module_name).parameters():
                        p.requires_grad = False
                    logger.log(f"[Controller] Froze module: {module_name}")

        if "unfreeze_modules" in batch_feedback:
            for module_name in batch_feedback["unfreeze_modules"]:
                if hasattr(slm, module_name):
                    for p in getattr(slm, module_name).parameters():
                        p.requires_grad = True
                    logger.log(f"[Controller] Unfroze module: {module_name}")

    # -------------------- Epoch-level Controller --------------------
    avg_epoch_loss = epoch_loss_total / max(num_batches, 1)  # avoid division by zero
    feedback = controller.control_stage(
        stage="epoch_end",
        context={"loss": avg_epoch_loss, "epoch": epoch+1},
    )
    logger.log(f"[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")

    logger.log("[Controller Feedback]")
    for key, value in feedback.items():
        logger.log(f"  - {key}: {value}")

    if feedback.get("stop_training", False):
        logger.log("[Controller] Stop requested, ending fine-tuning")
        break

    # Save checkpoint after each epoch
    ckpt_path = os.path.join(SAVE_PATH, f"slm_finetuned_epoch{epoch+1}.pt")
    torch.save(slm.state_dict(), ckpt_path)
    logger.log(f"Checkpoint saved: {ckpt_path}")

# -------------------- Save final fine-tuned model --------------------
torch.save(slm.state_dict(), os.path.join(SAVE_PATH, "slm_finetuned.pt"))
torch.save(memory.state_dict(), os.path.join(SAVE_PATH, "memory_finetuned.pt"))
logger.log("Fine-tuning completed and saved!")
