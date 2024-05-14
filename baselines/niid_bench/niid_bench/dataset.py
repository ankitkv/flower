"""Create the Ray Dataset refs."""

import ray
from flwr.common.typing import List, Optional, Tuple
from omegaconf import DictConfig
from ray import data as ray_data

from niid_bench.dataset_preparation import (
    partition_data,
    partition_data_dirichlet,
    partition_data_label_quantity,
)


# pylint: disable=too-many-locals, too-many-branches
def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,  # FIXME seed is not used everywhere yet
) -> Tuple[List[ray_data.Dataset], List[ray_data.Dataset], ray_data.Dataset]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoaders for training, validation, and testing.
    """
    partitioning = ""
    if "partitioning" in config:
        partitioning = config.partitioning
    # partition the data
    if partitioning == "dirichlet":
        alpha = 0.5
        if "alpha" in config:
            alpha = config.alpha
        counted_datasets, test_ds = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config.name,
            fill_zero_pixel=config.fill_zero_pixel,
        )
    elif partitioning == "label_quantity":
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config.labels_per_client
        counted_datasets, test_ds = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config.name,
            fill_zero_pixel=config.fill_zero_pixel,
        )
    elif partitioning == "iid":
        counted_datasets, test_ds = partition_data(
            num_clients,
            similarity=1.0,
            seed=seed,
            dataset_name=config.name,
            fill_zero_pixel=config.fill_zero_pixel,
        )
    elif partitioning == "iid_noniid":
        similarity = 0.5
        if "similarity" in config:
            similarity = config.similarity
        counted_datasets, test_ds = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config.name,
            fill_zero_pixel=config.fill_zero_pixel,
        )

    # split each partition into train/val and create DataLoader
    train_dss = []
    val_dss = []
    for i, (num_examples, dataset) in enumerate(counted_datasets):
        len_val = int(num_examples / (1 / val_ratio)) if val_ratio > 0 else 0
        if len_val > 0:
            ds_train, ds_val = dataset.train_test_split(
                len_val, shuffle=True, seed=seed + i
            )
        else:
            ds_train, ds_val = dataset, None
        train_dss.append(ray.put(ds_train))
        val_dss.append(ray.put(ds_val))

    # test_ds is not stored in Ray object store because we will only use it from
    # the driver.
    return train_dss, val_dss, test_ds
