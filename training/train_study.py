# train_core.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from training.model import init_core_slm
from utils.trainer_utils import freeze_base_model, save_checkpoint, set_global_seed

TOKENIZED_FILE = "data/tokenized/study_data_tokenized.ndjson"
CUSTOM_TOKENIZER_DIR = "models/tokenizer"
FALLBACK_TOKENIZER = "gpt2"
USE_ADAPTERS = True
ADAPTER_SIZE = 256
SEED = 42

def count_lines(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    set_global_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    if os.path.exists(CUSTOM_TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)

    # Init model
    model = init_core_slm(tokenizer, use_adapters=USE_ADAPTERS, adapter_size=ADAPTER_SIZE)
    if USE_ADAPTERS:
        freeze_base_model(model)
    model.to(device)

    # ✅ Load dataset in streaming mode (never loads into RAM fully)
    dataset = load_dataset(
        "json",
        data_files=TOKENIZED_FILE,
        split="train",
        streaming=True
    )

    # ✅ Split into train/eval
    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ✅ Use collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/core_slm",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-4 if USE_ADAPTERS else 5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    save_checkpoint(model, tokenizer, "checkpoints/study_slm/final")

if __name__ == "__main__":
    print("Total samples are training on:", count_lines(TOKENIZED_FILE))

    main()
