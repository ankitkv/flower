"""Defines the classes and functions for FedAvg."""

import os
from collections import OrderedDict
from logging import INFO

import numpy as np
import torch
import torch.nn as nn
import wandb
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from ray import ObjectRef
from ray import data as ray_data
from torch.optim import SGD, Optimizer

from niid_bench.client import FlowerClient


class ResumingFedAvg(FedAvg):
    """FedAvg wrapper that allows resuming and WandB logging."""

    def __init__(
        self,
        net_for_checkpoint: torch.nn.Module,
        device: torch.device,
        checkpoint_path: str,
        eval_every_rounds: int = 1,
        save_every_rounds: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, **kwargs
        )
        self.net_for_checkpoint = net_for_checkpoint
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.eval_every_rounds = eval_every_rounds
        self.save_every_rounds = save_every_rounds
        self.init_round = 0

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is None and os.path.exists(self.checkpoint_path):
            log(INFO, f"Resuming from {self.checkpoint_path} ...")
            loaded_round, loaded_state_dict = torch.load(
                self.checkpoint_path, map_location=self.device
            )
            self.net_for_checkpoint.load_state_dict(loaded_state_dict)
            state_dict_ndarrays = [
                v.cpu().numpy() for v in self.net_for_checkpoint.state_dict().values()
            ]
            initial_parameters = ndarrays_to_parameters(state_dict_ndarrays)
            self.init_round = loaded_round
            log(INFO, f"Resumed server parameters after round {self.init_round}.")
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if server_round % self.eval_every_rounds == 0:
            loss, metrics = super().evaluate(server_round, parameters)
            wandb.log({"round": server_round, "eval/loss": loss}, step=server_round)
            wandb.log(
                {f"eval/{k}": v for k, v in metrics.items()},
                step=server_round,
                commit=True,
            )
            return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        cids = [client.cid for client, _ in client_instructions]
        log(INFO, f"Sampled clients {cids}")
        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        # Call aggregate_fit from base class (FedAvg) to aggregate params and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            if server_round % self.save_every_rounds == 0:
                log(
                    INFO,
                    f"Saving round {server_round} aggregated_parameters to "
                    f"{self.checkpoint_path} ...",
                )

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(
                    self.net_for_checkpoint.state_dict().keys(), aggregated_ndarrays
                )
                state_dict = OrderedDict(
                    {k: torch.from_numpy(v) for k, v in params_dict}
                )
                self.net_for_checkpoint.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(
                    (server_round, self.net_for_checkpoint.state_dict()),
                    self.checkpoint_path,
                )
                log(INFO, "Saved server parameters.")

            aggregated_metrics["param_norm"] = np.linalg.norm(
                np.stack([np.linalg.norm(p) for p in aggregated_ndarrays])
            )

        if aggregated_metrics:
            log(INFO, f"aggregated fit metrics: {aggregated_metrics}")
            wandb.log(
                {f"fit/{k}": v for k, v in aggregated_metrics.items()},
                step=server_round,
            )
        return aggregated_parameters, aggregated_metrics


class FedAvgClient(FlowerClient):
    """Flower client implementing FedAvg."""

    def fit(self, ins: FitIns) -> FitRes:
        """Implement distributed fit function for a given client for FedAvg."""
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(ndarrays)
        loss, num_examples = train_fedavg(
            net=self.net,
            train_ds=self.train_ds,
            device=self.device,
            epochs=self.num_epochs,
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


def fit_metrics_aggregation_fn(
    client_metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate client metrics when training."""
    aggregated_metrics = {}
    total_samples = sum([num_samples for num_samples, _ in client_metrics])
    for num_samples, metrics in client_metrics:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * (num_samples / total_samples)
    return aggregated_metrics


def train_fedavg(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    epochs: int,
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
    net.train()
    loss = 0.0
    num_examples = 0
    for _ in range(epochs):
        loss_, num_examples_ = _train_one_epoch(
            net=net,
            train_ds=train_ds,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loss += loss_ * num_examples_
        num_examples += num_examples_
    return loss / num_examples, num_examples


def _train_one_epoch(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
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
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    drop_last: bool = False,
) -> Callable[[str], FedAvgClient]:
    """Generate the client function that creates the FedAvg flower clients.

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
    momentum : float
        The momentum for SGD optimizer of clients.
    weight_decay : float
        The weight decay for SGD optimizer of clients.

    Returns
    -------
    Callable[[str], FedAvgClient]
        The client function that creates the FedAvg flower clients.
    """

    def client_fn(cid: str) -> FedAvgClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different train_ds/val_ds, so each client
        # will train and evaluate on their own unique data
        train_ds = train_dss[int(cid)]
        val_ds = val_dss[int(cid)]

        return FedAvgClient(
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            momentum=momentum,
            weight_decay=weight_decay,
            drop_last=drop_last,
        )

    return client_fn
