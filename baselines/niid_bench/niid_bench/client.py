"""Base client."""

from collections import OrderedDict

import flwr as fl
import ray
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from ray import ObjectRef

from niid_bench.evaluate import test


class FlowerClient(fl.client.Client):
    """Base Flower client."""

    def __init__(
        self,
        *,
        train_ds: ObjectRef,
        val_ds: ObjectRef,
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        momentum: float,
        weight_decay: float,
        drop_last: bool,
    ) -> None:
        super().__init__()
        self._train_ds = train_ds
        self._val_ds = val_ds
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.drop_last = drop_last
        self._device = None
        self._net = None

    @property
    def train_ds(self):
        """Return this client's training dataset."""
        return ray.get(self._train_ds)

    @property
    def val_ds(self):
        """Return this client's validation dataset."""
        return ray.get(self._val_ds)

    @property
    def device(self):
        """Return the device."""
        if self._device is None:
            self._device = self.context.device
        return self._device

    @property
    def net(self):
        """Return the local model."""
        if self._net is None:
            self._net = self.context.net
        return self._net

    def get_ndarrays(self):
        """Return ndarrays of the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters."""
        ndarrays = self.get_ndarrays()
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def set_parameters(self, ndarrays):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate using given parameters."""
        ndarrays = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(ndarrays)
        loss, acc, num_examples = test(
            self.net, self.val_ds, self.device, batch_size=self.batch_size
        )

        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples,
            metrics={"accuracy": float(acc)},
        )
