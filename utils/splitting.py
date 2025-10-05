def stream_train_test_split(dataset, test_size=0.1, total_samples=None, seed=42):
    """
    Perform a memory-safe train/test split on a streaming dataset.

    Args:
        dataset (datasets.IterableDataset): streaming dataset
        test_size (float): fraction of samples to allocate to test set
        total_samples (int): total number of lines in dataset (counted beforehand)
        seed (int): random seed for reproducibility

    Returns:
        (train_gen_fn, eval_gen_fn): two generator functions (not generator objects)
    """
    assert 0 < test_size < 1, "test_size must be a float in (0,1)"
    import random
    rng = random.Random(seed)

    if total_samples is not None:
        target_eval = int(total_samples * test_size)
    else:
        target_eval = None

    def train_gen_fn():
        eval_count = 0
        for sample in dataset:
            if target_eval is not None and eval_count >= target_eval:
                yield sample
                continue
            if rng.random() < test_size:
                eval_count += 1
                continue
            yield sample

    def eval_gen_fn():
        eval_count = 0
        for sample in dataset:
            if target_eval is not None and eval_count >= target_eval:
                continue
            if rng.random() < test_size:
                eval_count += 1
                yield sample
            else:
                continue

    return train_gen_fn, eval_gen_fn  # return functions, not objects
