"""Evaluation functions."""

from collections import OrderedDict

import torch
import torch.nn as nn
from flwr.common import Scalar
from flwr.common.typing import Callable, Dict, NDArrays, Optional, Tuple
from ray import data as ray_data


def gen_evaluate_fn(
    test_ds: ray_data.Dataset,
    net_for_eval: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    test_ds : ray_data.Dataset
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    batch_size : int
        Batch size per client.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        params_dict = zip(net_for_eval.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net_for_eval.load_state_dict(state_dict, strict=True)
        loss, accuracy, _ = test(
            net_for_eval, test_ds, device=device, batch_size=batch_size
        )
        return loss, {"accuracy": accuracy}

    return evaluate


def test(
    net: nn.Module, test_ds: ray_data.Dataset, device: torch.device, batch_size: int
) -> Tuple[float, float]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    test_ds : ray_data.Dataset
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.
    batch_size : int
        The batch size for evaluation.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in test_ds.iter_torch_batches(batch_size=batch_size, device=device):
            data, target = batch["img"], batch["label"]
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc, total
