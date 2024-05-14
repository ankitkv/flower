"""Entry point for the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc.
"""

import os

import flwr as fl
import hydra
import ray
import torch
import wandb
from flwr.server.client_manager import SimpleClientManager
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from niid_bench.actor import ModelClientAppActor
from niid_bench.dataset import load_datasets
from niid_bench.evaluate import gen_evaluate_fn
from niid_bench.methods.fednova import FedNovaServer, FedNovaStrategy
from niid_bench.methods.scaffold import ScaffoldServer, ScaffoldStrategy
from niid_bench.server import ResumingServer


@hydra.main(config_path="conf", config_name="fedavg", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print parsed config
    if cfg.model.num_classes is None:
        if cfg.dataset_name == "cifar10":
            cfg.model.num_classes = 10
        elif cfg.dataset_name == "cifar100":
            cfg.model.num_classes = 100
        else:
            raise NotImplementedError
    print(OmegaConf.to_yaml(cfg, resolve=True))
    log_dir = os.path.join(cfg.log_dir, cfg.name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, "model_latest.pth")

    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=log_dir,
        **cfg.wandb,
    )

    # Initialize Ray
    if cfg.ram_gb is not None:
        # Monkey-patch to let Ray decide optimal defaults based on the memory we actually
        # have allocated rather than the full system memory.
        get_system_memory = ray._private.utils.get_system_memory
        estimate_available_memory = ray._private.utils.estimate_available_memory
        ram_bytes = int(cfg.ram_gb * (1024**3))
        ray._private.utils.get_system_memory = lambda: ram_bytes
        ray._private.utils.estimate_available_memory = lambda: ram_bytes
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": cfg.ray_dashboard,
        "log_to_driver": cfg.log_to_driver,
        "num_cpus": cfg.num_cpus,
        "num_gpus": cfg.num_gpus,
    }
    ray.init(**ray_init_args)
    if cfg.ram_gb is not None:
        ray._private.utils.get_system_memory = get_system_memory
        ray._private.utils.estimate_available_memory = estimate_available_memory

    # Prepare dataset
    train_dss, val_dss, test_ds = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.val_split,
    )

    # Define clients
    client_fn = None
    # pylint: disable=protected-access
    if cfg.client_fn._target_ == "niid_bench.methods.scaffold.gen_client_fn":
        client_cv_dir = os.path.join(log_dir, "client_cvs")
        print("Local cvs for scaffold clients are saved to: ", client_cv_dir)
        client_fn = call(
            cfg.client_fn,
            train_dss,
            val_dss,
            client_cv_dir=client_cv_dir,
        )
    else:
        client_fn = call(
            cfg.client_fn,
            train_dss,
            val_dss,
        )

    device = torch.device(cfg.server_device if torch.cuda.is_available() else "cpu")
    global_model = instantiate(cfg.model).to(device)

    evaluate_fn = gen_evaluate_fn(
        test_ds, net_for_eval=global_model, device=device, batch_size=cfg.batch_size
    )

    # Define strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        net_for_checkpoint=global_model,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    # Define server
    if isinstance(strategy, FedNovaStrategy):
        server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
    elif isinstance(strategy, ScaffoldStrategy):
        server = ScaffoldServer(strategy=strategy, client_manager=SimpleClientManager())
    else:
        server = ResumingServer(strategy=strategy, client_manager=SimpleClientManager())

    # Start simulation
    history = fl.simulation.start_simulation(
        actor_type=ModelClientAppActor,
        actor_kwargs={"model": ray.put(cfg.model)},
        num_clients=cfg.num_clients,
        client_fn=client_fn,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        server=server,
        ray_init_args=ray_init_args,
        keep_initialised=True,
    )
    print(history)


if __name__ == "__main__":
    main()
