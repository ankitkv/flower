"""Download data and partition data with different partitioning strategies."""

import datasets
import numpy as np
import ray
import torch
from flwr.common.typing import Callable, Dict, List, NDArray, Optional, Tuple
from ray import data as ray_data
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import v2
from torchvision.transforms.v2 import _utils as v2_utils


class BatchedRandomCrop(v2.RandomCrop):
    """Based on torchvision.transforms.v2.RandomCrop to support batched random crop."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform the batched random crop."""
        padded_height, padded_width = inputs.shape[-2:]

        if self.padding is not None:
            pad_left, pad_right, pad_top, pad_bottom = self.padding
            padded_height += pad_top + pad_bottom
            padded_width += pad_left + pad_right
        else:
            pad_left = pad_right = pad_top = pad_bottom = 0

        cropped_height, cropped_width = self.size

        if self.pad_if_needed:
            if padded_height < cropped_height:
                diff = cropped_height - padded_height
                pad_top += diff
                pad_bottom += diff
                padded_height += 2 * diff

            if padded_width < cropped_width:
                diff = cropped_width - padded_width
                pad_left += diff
                pad_right += diff
                padded_width += 2 * diff

        if padded_height < cropped_height or padded_width < cropped_width:
            raise ValueError(
                f"Required crop size {(cropped_height, cropped_width)} is larger than "
                f"{'padded ' if self.padding is not None else ''}input image size "
                f"{(padded_height, padded_width)}."
            )

        # We need a different order here than we have in self.padding since this padding
        # will be parsed again in `F.pad`
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        needs_pad = any(padding)

        needs_crop = False
        if padded_height > cropped_height:
            needs_crop = True
            top = torch.randint(
                0,
                padded_height - cropped_height + 1,
                size=inputs.shape[:1],
                device=inputs.device,
            )
        else:
            top = inputs.new_zeros(inputs.shape[0], dtype=torch.long)
        if padded_width > cropped_width:
            needs_crop = True
            left = torch.randint(
                0,
                padded_width - cropped_width + 1,
                size=inputs.shape[:1],
                device=inputs.device,
            )
        else:
            left = inputs.new_zeros(inputs.shape[0], dtype=torch.long)

        outs = inputs

        if needs_pad:
            fill = v2_utils._get_fill(self._fill, type(outs))
            outs = self._call_kernel(
                v2.functional.pad,
                outs,
                padding=padding,
                fill=fill,
                padding_mode=self.padding_mode,
            )

        if needs_crop:
            rows = (
                torch.arange(cropped_height, dtype=torch.long, device=outs.device)
                + top[:, None]
            )
            columns = (
                torch.arange(cropped_width, dtype=torch.long, device=outs.device)
                + left[:, None]
            )
            outs = outs.permute(1, 0, 2, 3)
            outs = outs[
                :,
                torch.arange(inputs.shape[0])[:, None, None],
                rows[:, torch.arange(cropped_height), None],
                columns[:, None],
            ]
            outs = outs.permute(1, 0, 2, 3)

        return outs


class BatchedRandomHorizontalFlip(v2.Transform):
    """Batched random flip."""

    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                "`p` should be a floating point value in the interval [0.0, 1.0]."
            )
        super().__init__()
        self.p = p

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform the batched random flip."""
        inputs = inputs.clone()
        do_flip = torch.rand(inputs.shape[0]) < self.p
        inputs[do_flip] = inputs[do_flip].flip(-1)
        return inputs


def _download_data(
    dataset_name="cifar10", fill_zero_pixel=True
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Download the requested dataset. Currently supports cifar10 and cifar100.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    if dataset_name == "cifar10":
        train_split, test_split = "train", "test"
        img_column, label_column = "img", "label"
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        normalize_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std),
            ]
        )
        if fill_zero_pixel:
            fill = tuple((-np.array(mean) / np.array(std)).tolist())
        else:
            fill = 0
        augment_transform = v2.Compose(
            [
                BatchedRandomCrop(32, padding=4, fill=fill),
                BatchedRandomHorizontalFlip(),
            ]
        )
    elif dataset_name == "cifar100":
        train_split, test_split = "train", "test"
        img_column, label_column = "img", "fine_label"
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        normalize_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std),
            ]
        )
        if fill_zero_pixel:
            fill = tuple((-np.array(mean) / np.array(std)).tolist())
        else:
            fill = 0
        augment_transform = v2.Compose(
            [
                BatchedRandomCrop(32, padding=4, fill=fill),
                BatchedRandomHorizontalFlip(),
            ]
        )

    ds = datasets.load_dataset(dataset_name)
    ds = ds.remove_columns(
        [c for c in ds[train_split].features if c not in [img_column, label_column]]
    )
    if img_column != "img":
        ds = ds.rename_column(img_column, "img")
    if label_column != "label":
        ds = ds.rename_column(label_column, "label")
    ds = ds.cast_column("img", datasets.Image(decode=False, mode="RGB"))

    return ds[train_split], ds[test_split], normalize_transform, augment_transform


def batch_decode(
    batch: Dict[str, NDArray],
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
) -> Dict[str, NDArray]:
    """Decode the images and apply an optional batch transformation."""
    image_decoder = datasets.Image(decode=True, mode="RGB")
    images = torch.stack(
        [v2.functional.to_image(image_decoder.decode_example(i)) for i in batch["img"]]
    )
    if transform is not None:
        images = transform(images)
    return {"img": images.numpy(), "label": batch["label"]}


def batch_augment(
    batch: Dict[str, NDArray],
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]],
) -> Dict[str, NDArray]:
    """Augment the images with a batch transformation."""
    return {
        "img": transform(torch.from_numpy(batch["img"])).numpy(),
        "label": batch["label"],
    }


# pylint: disable=too-many-locals
def partition_data(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10", fill_zero_pixel=True
) -> Tuple[List[Tuple[int, ray_data.Dataset]], ray_data.Dataset]:
    """Partition the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    raise NotImplementedError  # TODO only partition_data_label_quantity is implemented
    trainset, testset = _download_data(dataset_name, fill_zero_pixel=fill_zero_pixel)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset


def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10", fill_zero_pixel=True
) -> Tuple[List[Tuple[int, ray_data.Dataset]], ray_data.Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    raise NotImplementedError  # TODO only partition_data_label_quantity is implemented
    trainset, testset = _download_data(dataset_name, fill_zero_pixel=fill_zero_pixel)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def partition_data_label_quantity(
    num_clients,
    labels_per_client,
    seed=42,
    dataset_name="cifar10",
    fill_zero_pixel=True,
) -> Tuple[List[Tuple[int, ray_data.Dataset]], ray_data.Dataset]:
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset, normalize_transform, augment_transform = _download_data(
        dataset_name, fill_zero_pixel=fill_zero_pixel
    )
    prng = np.random.default_rng(seed)

    targets = trainset["label"]
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients_flat: List = []
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients_flat.extend(idx_k_split[ids].tolist())
                ids += 1
    trainset = trainset.select(idx_clients_flat)

    # TODO move the following to a separate function
    trainsets_per_client = (
        ray_data.from_huggingface(trainset)
        .map_batches(
            batch_decode,
            fn_kwargs={"transform": normalize_transform},
            zero_copy_batch=True,
        )
        .split(num_clients, equal=True)
    )
    testset = ray_data.from_huggingface(testset).map_batches(
        batch_decode, fn_kwargs={"transform": normalize_transform}, zero_copy_batch=True
    )

    # add counts and augmentations
    counted_trainsets_per_client = [
        (
            ts.count(),
            ts.map_batches(
                batch_augment,
                fn_kwargs={"transform": augment_transform},
                zero_copy_batch=True,
            ),
        )
        for ts in trainsets_per_client
    ]
    return counted_trainsets_per_client, testset


if __name__ == "__main__":
    ray_init_args = {
        "log_to_driver": False,
        "num_cpus": 4,
        "num_gpus": 1,
        "include_dashboard": False,
        "object_store_memory": int(0.3 * 16 * 1000 * 1024 * 1024),
        "_memory": int(0.7 * 16 * 1000 * 1024 * 1024),
    }
    ray.init(**ray_init_args)

    counted_trainsets_per_client, testset = partition_data_label_quantity(100, 1)
    for trial in range(4):
        print(f"Trial {trial}:")
        counts = []
        client_labels = []
        images = []
        labels = []
        for count, ts in counted_trainsets_per_client:
            counts.append(count)
            c_labels = set()
            for batch in ts.iter_torch_batches(
                batch_size=64, local_shuffle_buffer_size=6400, drop_last=False
            ):
                data, target = batch["img"], batch["label"]
                images.append(data)
                labels.append(target)
                c_labels.update(target.tolist())
            client_labels.append(c_labels)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        print(f"    Counts: {counts}")
        print(f"    Client labels: {client_labels}")
        for d in range(3):
            means = ", ".join(
                [f"{i}: {images[labels==i, d].mean().item():.2f}" for i in range(10)]
            )
            stds = ", ".join(
                [f"{i}: {images[labels==i, d].std().item():.2f}" for i in range(10)]
            )
            print(f"    Means ({d}): {{{means}}}")
            print(f"    Stds  ({d}): {{{stds}}}")
