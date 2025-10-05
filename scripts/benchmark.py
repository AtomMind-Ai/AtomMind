"""
Runs simple evaluation on trained model using sample study tasks.
Supports QA, summarization, and math problem answering.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate(model_path="models/core_slm_2b", task="qa", prompt="What is the capital of France?"):
    """
    Run a simple benchmark with a prompt.

    Args:
        model_path (str): Path to trained model checkpoint.
        task (str): Task type (qa, summarization, math).
        prompt (str): Input query.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    print("🔹 Prompt:", prompt)
    print("🔹 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    evaluate()
