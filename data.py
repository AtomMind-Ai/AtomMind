"""
Standalone Dataset Generator for AtomMind Training
- Uses existing loader module to generate & save dataset
- Appends new samples to datasets/dataset.txt if file exists
"""

from data import loader  # your existing loader.py
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for AtomMind training")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )
    args = parser.parse_args()

    # Generate dataset (appends to existing file)
    new_data = loader.generate_dataset(num_samples=args.num_samples)
    print(f"[Info] Generated {len(new_data)} new samples and saved to {loader.DATASET_PATH}")
