"""Actor that keeps a model in its memory."""

import ray
import torch
from flwr.client.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.common import Context, Message
from flwr.common.typing import Callable, Optional, Tuple
from flwr.simulation.ray_transport.ray_actor import VirtualClientEngineActor
from hydra.utils import instantiate
from omegaconf import DictConfig


@ray.remote
class ModelClientAppActor(VirtualClientEngineActor):
    """A Ray Actor class that runs client runs and keeps an instantiated model.

    Parameters
    ----------
    on_actor_init_fn: Optional[Callable[[], None]] (default: None)
        A function to execute upon actor initialization.
    """

    def __init__(
        self,
        on_actor_init_fn: Optional[Callable[[], None]] = None,
        model: DictConfig = None,
    ) -> None:
        super().__init__()
        if on_actor_init_fn:
            on_actor_init_fn()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = instantiate(model).to(self.device)

    def run(
        self,
        client_app_fn: Callable[[], ClientApp],
        message: Message,
        cid: str,
        context: Context,
    ) -> Tuple[str, Message, Context]:
        """Run a client run."""
        # Pass message through ClientApp and return a message
        # return also cid which is needed to ensure results
        # from the pool are correctly assigned to each ClientProxy
        try:
            # Load app
            app: ClientApp = client_app_fn()

            # Add model and device references to context
            context.device = self.device
            context.net = self.net

            # Handle task message
            out_message = app(message=message, context=context)

        except LoadClientAppError as load_ex:
            raise load_ex

        except Exception as ex:
            raise ClientAppException(str(ex)) from ex

        return cid, out_message, context
