"""Defines the classes and functions for SCAFFOLD."""

# TODO FIXME [Bug] Currently, this method produces a flat accuracy curve on CIFAR10.

import os
from logging import DEBUG, INFO, WARNING

import torch
import torch.nn as nn
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
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import FitResultsAndFailures, fit_clients
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate
from ray import ObjectRef
from ray import data as ray_data
from torch.optim import SGD

from niid_bench.client import FlowerClient
from niid_bench.methods.fedavg import ResumingFedAvg
from niid_bench.server import ResumingServer


class ScaffoldStrategy(ResumingFedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )


class ScaffoldServer(ResumingServer):
    """Implement server for SCAFFOLD."""

    def __init__(
        self,
        strategy: Strategy,
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.server_cv: List[torch.Tensor] = []

    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            # log(INFO, "Using initial parameters provided by strategy")
            # return parameters
            raise NotImplementedError("Resuming is not supported with SCAFFOLD yet.")

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=server_round
        )
        log(INFO, "Received initial parameters from one random client")
        self.server_cv = [
            torch.from_numpy(t)
            for t in parameters_to_ndarrays(get_parameters_res.parameters)
        ]
        return get_parameters_res.parameters

    # pylint: disable=too-many-locals
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
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
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

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(
                aggregated_result[0]
            )
        aggregated_parameters = aggregated_result_arrays_combined[
            : len(aggregated_result_arrays_combined) // 2
        ]
        aggregated_cv_update = aggregated_result_arrays_combined[
            len(aggregated_result_arrays_combined) // 2 :
        ]

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # update parameters x = x + 1* aggregated_update
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)


class ScaffoldClient(FlowerClient):
    """Flower client implementing scaffold."""

    def __init__(self, cid: int, save_dir: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.cid = cid
        self.client_cv = []
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def fit(self, ins: FitIns) -> FitRes:
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        parameters = parameters_to_ndarrays(ins.parameters)
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            # FIXME This does not work well with checkpointing!
            #       Use state: RecordSet = self.context.state for stateful clients
            self.client_cv = torch.load(
                f"{self.dir}/client_cv_{self.cid}.pt", map_location=self.device
            )
        # convert the server control variate to a list of tensors
        server_cv = [torch.from_numpy(cv).to(device=self.device) for cv in server_cv]
        loss, num_examples = train_scaffold(
            net=self.net,
            train_ds=self.train_ds,
            device=self.device,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            server_cv=server_cv,
            client_cv=self.client_cv,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        x = parameters
        y_i = self.get_ndarrays()
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_j = c_i_j.cpu().numpy()
            c_j = c_j.cpu().numpy()
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * num_examples))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append(y_i_j - x_j)
            server_update_c.append(c_i_n[-1] - c_i_j)
        self.client_cv = [torch.from_numpy(c) for c in c_i_n]
        # FIXME This does not work well with checkpointing! See above.
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c
        combined_updates_p = ndarrays_to_parameters(combined_updates)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=combined_updates_p,
            num_examples=num_examples,
            metrics={"loss": loss},
        )


class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, grads, step_size, momentum, weight_decay):
        super().__init__(
            grads, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        """Implement the custom step function fo SCAFFOLD."""
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])


def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Extend the list of parameters with the server control variate."""
    # extend the list of parameters arrays with the cv arrays
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    parameters_np.extend(cv_np)
    return ndarrays_to_parameters(parameters_np)


def train_scaffold(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
    batch_size: int,
    drop_last: bool,
) -> Tuple[float, int]:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using SCAFFOLD.

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
    server_cv : torch.Tensor
        The server's control variate.
    client_cv : torch.Tensor
        The client's control variate.
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
    optimizer = ScaffoldOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    net.train()
    loss = 0.0
    num_examples = 0
    for _ in range(epochs):
        loss_, num_examples_ = _train_one_epoch_scaffold(
            net=net,
            train_ds=train_ds,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            server_cv=server_cv,
            client_cv=client_cv,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loss += loss_ * num_examples_
        num_examples += num_examples_
    return loss / num_examples, num_examples


def _train_one_epoch_scaffold(
    net: nn.Module,
    train_ds: ray_data.Dataset,
    device: torch.device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
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
        optimizer.step_custom(server_cv, client_cv)

    return total_loss / num_examples, num_examples


def gen_client_fn(
    train_dss: List[ObjectRef],
    val_dss: List[ObjectRef],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    client_cv_dir: str,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    drop_last: bool = False,
) -> Callable[[str], ScaffoldClient]:
    """Generate the client function that creates the scaffold flower clients.

    Parameters
    ----------
    train_dss: List[ObjectRef]
        A list of Ray Dataset refs, each pointing to the dataset training partition
        belonging to a particular client.
    val_dss: List[ObjectRef]
        A list of Ray Dataset refs, each pointing to the dataset validation partition
        belonging to a particular client.
    client_cv_dir : str
        The directory where the client control variates are stored (persistent storage).
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
    Callable[[str], ScaffoldClient]
        The client function that creates the scaffold flower clients.
    """

    def client_fn(cid: str) -> ScaffoldClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different train_ds/val_ds, so each client
        # will train and evaluate on their own unique data
        train_ds = train_dss[int(cid)]
        val_ds = val_dss[int(cid)]

        return ScaffoldClient(
            cid=int(cid),
            train_ds=train_ds,
            val_ds=val_ds,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            save_dir=client_cv_dir,
            momentum=momentum,
            weight_decay=weight_decay,
            drop_last=drop_last,
        )

    return client_fn
