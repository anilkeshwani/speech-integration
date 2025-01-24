def compute_iterations_per_epoch(dataset_size: int, global_batch_size: int, drop_last: bool) -> int:
    return dataset_size // global_batch_size if drop_last else -(-dataset_size // global_batch_size)


def compute_n_epochs(iterations: int, dataset_size: int, global_batch_size: int, drop_last: bool) -> int:
    iterations_per_epoch = compute_iterations_per_epoch(dataset_size, global_batch_size, drop_last)
    return -(-iterations // iterations_per_epoch)
