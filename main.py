"""
main.py

Final entrypoint for running the StudyHelper-SLM system.
Loads Core + StudyNet models, connects Memory, and uses Orchestrator to answer queries.
"""

import torch
from orchestrator.orchestrator import Orchestrator
from memory.memory_system import MemorySystem
from utils.tokenizer_utils import get_tokenizer
from transformers import AutoModelForCausalLM


def load_models():
    """Load Core and StudyNet models with shared tokenizer."""
    tokenizer = get_tokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Loading models on {device}...")

    core_model = AutoModelForCausalLM.from_pretrained("models/core/final").to(device)
    study_model = AutoModelForCausalLM.from_pretrained("models/study/final").to(device)

    core_model.eval()
    study_model.eval()

    return tokenizer, core_model, study_model, device


def run_system():
    # Load models + tokenizer
    tokenizer, core_model, study_model, device = load_models()

    # Init memory
    memory = MemorySystem()

    # Init orchestrator
    orchestrator = Orchestrator(core_model, study_model, memory, tokenizer, device=device)

    print("✅ StudyHelper-SLM System Ready!\n")

    # Simple REPL loop
    while True:
        query = input("👤 You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = orchestrator.handle_query(query)
        print(f"🤖 Assistant: {response}\n")


if __name__ == "__main__":
    run_system()
