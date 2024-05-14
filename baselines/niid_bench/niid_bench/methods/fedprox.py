"""Defines the classes and functions for FedProx."""

import torch
import torch.nn as nn
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    Status,
    parameters_to_ndarrays,
)
from flwr.common.typing import Callable, List, Tuple
from ray import ObjectRef
from ray import data as ray_data
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer

from niid_bench.client import FlowerClient


class FedProxClient(FlowerClient):
    """Flower client implementing FedProx."""

    def __init__(self, proximal_mu: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu

    def fit(self, ins: FitIns) -> FitRes:
        """Implement distributed fit function for a given client for FedProx."""
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(ndarrays)
        loss, num_examples = train_fedprox(
            net=self.net,
            train_ds=self.train_ds,
            device=self.device,
            epochs=self.num_epochs,
            proximal_mu=self.proximal_mu,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        final_p = self.get_parameters(GetParametersIns(config={})).parameters

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=final_p,
            num_examples=num_examples,
            metrics={"loss": loss},
        )


def train_fedprox(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    epochs: int,
    proximal_mu: float,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    drop_last: bool,
) -> Tuple[float, int]:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedAvg.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    train_ds : ray_data.Dataset
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    proximal_mu : float
        The proximal mu parameter.
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.
    batch_size : int
        The batch size for training.
    drop_last : bool
        Whether to drop the last batch if it cannot be filled.

    Returns
    -------
    float
        Training loss.
    """
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    global_params = [param.detach().clone() for param in net.parameters()]
    net.train()
    loss = 0.0
    num_examples = 0
    for _ in range(epochs):
        loss_, num_examples_ = _train_one_epoch_fedprox(
            net=net,
            global_params=global_params,
            train_ds=train_ds,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            proximal_mu=proximal_mu,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loss += loss_ * num_examples_
        num_examples += num_examples_
    return loss / num_examples, num_examples


def _train_one_epoch_fedprox(
    net: nn.Module,
    global_params: List[Parameter],
    train_ds: ray_data.Dataset,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    proximal_mu: float,
    batch_size: int,
    drop_last: bool,
) -> Tuple[float, int]:
    """Train the network on the training set for one epoch."""
    total_loss = 0.0
    num_examples = 0
    for batch in train_ds.iter_torch_batches(
        batch_size=batch_size,
        local_shuffle_buffer_size=batch_size * 100,
        drop_last=drop_last,
        device=device,
    ):
        optimizer.zero_grad()
        data, target = batch["img"], batch["label"]
        output = net(data)
        loss = criterion(output, target)
        proximal_term = 0.0
        for param, global_param in zip(net.parameters(), global_params):
            proximal_term += torch.norm(param - global_param) ** 2
        loss += (proximal_mu / 2) * proximal_term
        loss.backward()
        total_loss += loss.item() * target.size(0)
        num_examples += target.size(0)
        optimizer.step()

    return total_loss / num_examples, num_examples


def gen_client_fn(
    train_dss: List[ObjectRef],
    val_dss: List[ObjectRef],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    proximal_mu: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    drop_last: bool = False,
) -> Callable[[str], FedProxClient]:
    """Generate the client function that creates the FedProx flower clients.

    Parameters
    ----------
    train_dss: List[ObjectRef]
        A list of Ray Dataset refs, each pointing to the dataset training partition
        belonging to a particular client.
    val_dss: List[ObjectRef]
        A list of Ray Dataset refs, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    proximal_mu : float
        The proximal mu parameter.
    momentum : float
        The momentum for SGD optimizer of clients.
    weight_decay : float
        The weight decay for SGD optimizer of clients.

    Returns
    -------
    Callable[[str], FedProxClient]
        The client function that creates the FedProx flower clients
    """

    def client_fn(cid: str) -> FedProxClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different train_ds/val_ds, so each client
        # will train and evaluate on their own unique data
        train_ds = train_dss[int(cid)]
        val_ds = val_dss[int(cid)]

        return FedProxClient(
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            proximal_mu=proximal_mu,
            momentum=momentum,
            weight_decay=weight_decay,
            drop_last=drop_last,
        )

    return client_fn
