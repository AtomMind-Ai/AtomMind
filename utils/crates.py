# Proper generator wrappers
def train_gen(stream):
    """
    Generator function for the training dataset.
    
    Yields:
        dict: A single tokenized training example (with input_ids, attention_mask, labels, etc.)
    Notes:
        - Wraps the `stream` iterator.
        - Allows Hugging Face `IterableDataset.from_generator` to consume samples lazily.
        - Combined with `.shuffle(buffer_size=...)` to add randomness without loading entire dataset into memory.
    """
    for ex in stream:
        yield ex


def eval_gen(stream):
    """
    Generator function for the evaluation dataset.
    
    Yields:
        dict: A single tokenized evaluation example.
    Notes:
        - Wraps the `stream` iterator.
        - Evaluation dataset is kept deterministic (no shuffling).
        - Ensures reproducible evaluation during training.
    """
    for ex in stream:
        yield ex
