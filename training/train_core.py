# train_core.py

import os
import torch
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from training.model import init_core_slm
from utils.trainer_utils import freeze_base_model, save_checkpoint, set_global_seed
from utils.splitting import stream_train_test_split

TOKENIZED_FILE = "data/tokenized/core_data_tokenized.ndjson"
CUSTOM_TOKENIZER_DIR = "models/tokenizer"
FALLBACK_TOKENIZER = "gpt2"
USE_ADAPTERS = True
ADAPTER_SIZE = 256
SEED = 42

def count_lines(file_path: str) -> int:
    """Count total samples without loading into memory"""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    set_global_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔹 Load tokenizer
    if os.path.exists(CUSTOM_TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)

    # 🔹 Init model
    model = init_core_slm(tokenizer, use_adapters=USE_ADAPTERS, adapter_size=ADAPTER_SIZE)
    if USE_ADAPTERS:
        freeze_base_model(model)
    model.to(device)

    # 🔹 Count total samples
    total_samples = count_lines(TOKENIZED_FILE)
    print(f"Total samples available: {total_samples}")

    # 🔹 Load dataset in streaming mode
    dataset = load_dataset(
        "json",
        data_files=TOKENIZED_FILE,
        split="train",
        streaming=True
    )

    # 🔹 Manual train/test split
    train_gen_fn, eval_gen_fn = stream_train_test_split(
        dataset,
        test_size=0.05,
        total_samples=total_samples,
        seed=SEED,
    )

    train_dataset = IterableDataset.from_generator(
        train_gen_fn
    ).shuffle(buffer_size=10_000, seed=SEED)

    eval_dataset = IterableDataset.from_generator(
        eval_gen_fn
    )


    # 🔹 Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 🔹 Training arguments
    # Compute training steps manually
    batch_size = 2 * 8  # per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = total_samples // batch_size
    max_steps = steps_per_epoch * 3  # since you want 3 epochs

    training_args = TrainingArguments(
        output_dir="checkpoints/core_slm",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-4 if USE_ADAPTERS else 5e-5,
        fp16=torch.cuda.is_available(),
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=steps_per_epoch,  # save once per epoch
        eval_strategy="steps",
        eval_steps=steps_per_epoch,  # eval once per epoch
    )


    # 🔹 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    save_checkpoint(model, tokenizer, "checkpoints/core_slm/final")

if __name__ == "__main__":
    main()
