"""Defines the classes and functions for FedNova."""

from functools import reduce
from logging import DEBUG, INFO, WARNING

import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import (
    Callable,
    Dict,
    List,
    NDArrays,
    Optional,
    Scalar,
    Tuple,
    Union,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import FitResultsAndFailures, fit_clients
from ray import ObjectRef
from ray import data as ray_data
from torch.optim import SGD, Optimizer

from niid_bench.client import FlowerClient
from niid_bench.methods.fedavg import ResumingFedAvg
from niid_bench.server import ResumingServer


class FedNovaStrategy(ResumingFedAvg):
    """Custom FedAvg strategy with fednova based configuration and aggregation."""

    def aggregate_fit_custom(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c_fact = sum(
            [
                float(fit_res.metrics["a_i"]) * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c_fact * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        # Aggregate grad updates, t_eff*(sum_i(p_i*\eta*d_i))
        grad_updates_aggregated = aggregate_fednova(new_weights_results)
        # Final parameters = server_params - grad_updates_aggregated
        aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


class FedNovaServer(ResumingServer):
    """Implement server for FedNova."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[FedNovaStrategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FedNovaStrategy = (
            strategy if strategy is not None else FedNovaStrategy()
        )

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        params_np = parameters_to_ndarrays(self.parameters)
        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit_custom(
            server_round, params_np, results, failures
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)


class FedNovaClient(FlowerClient):
    """Flower client implementing FedNova."""

    def fit(self, ins: FitIns) -> FitRes:
        """Implement distributed fit function for a given client for FedNova."""
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(ndarrays)
        a_i, g_i, num_examples = train_fednova(
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
        # final_p_np = self.get_parameters(GetParametersIns(config={})).parameters
        g_i_np = [param.cpu().numpy() for param in g_i]
        g_i_p = ndarrays_to_parameters(g_i_np)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=g_i_p,
            num_examples=num_examples,
            metrics={"a_i": a_i},
        )


def aggregate_fednova(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Implement custom aggregate function for FedNova."""
    # Create a list of weights, each multiplied by the weight_factor
    weighted_weights = [
        [layer * factor for layer in weights] for weights, factor in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def train_fednova(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    drop_last: bool,
) -> Tuple[float, List[torch.Tensor], int]:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedNova.

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
    tuple[float, List[torch.Tensor]]
        The a_i and g_i values.
    """
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    local_steps = 0
    num_examples = 0
    # clone all the parameters
    prev_net = [param.detach().clone() for param in net.parameters()]
    for _ in range(epochs):
        net, local_steps, num_examples_ = _train_one_epoch_fednova(
            net=net,
            train_ds=train_ds,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            local_steps=local_steps,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        num_examples += num_examples_
    # compute ||a_i||_1
    a_i = (
        local_steps - (momentum * (1 - momentum**local_steps) / (1 - momentum))
    ) / (1 - momentum)
    # compute g_i
    g_i = [
        torch.div(prev_param - param.detach(), a_i)
        for prev_param, param in zip(prev_net, net.parameters())
    ]

    return a_i, g_i, num_examples


def _train_one_epoch_fednova(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    local_steps: int,
    batch_size: int,
    drop_last: bool,
) -> Tuple[nn.Module, int, int]:
    """Train the network on the training set for one epoch."""
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
        num_examples += target.size(0)
        optimizer.step()
        local_steps += 1
    return net, local_steps, num_examples


def gen_client_fn(
    train_dss: List[ObjectRef],
    val_dss: List[ObjectRef],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    drop_last: bool = False,
) -> Callable[[str], FedNovaClient]:
    """Generate the client function that creates the FedNova flower clients.

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
    Callable[[str], FedNovaClient]
        The client function that creates the FedNova flower clients.
    """

    def client_fn(cid: str) -> FedNovaClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different train_ds/val_ds, so each client
        # will train and evaluate on their own unique data
        train_ds = train_dss[int(cid)]
        val_ds = val_dss[int(cid)]

        return FedNovaClient(
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
