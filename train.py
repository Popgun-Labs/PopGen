import hydra
from omegaconf import DictConfig

from popgen.setup import setup_worker, setup_loaders


@hydra.main(config_path="config/config.yaml")
def train(cfg: DictConfig) -> None:
    # get the experiment name
    name = cfg.get("name", False)
    if not name:
        raise Exception("Must specify experiment name on CLI. e.g. `python train.py name=vae ...`")

    # setup the worker
    overwrite = cfg.get("overwrite", False)
    worker, cfg = setup_worker(name, cfg, overwrite=overwrite)

    # setup data loaders
    train_loader, test_loader = setup_loaders(
        dataset_class=cfg["dataset_class"], data_opts=cfg["dataset"], loader_opts=cfg["loader"]
    )

    # train
    worker.run(train_loader, test_loader, cfg["nb_epoch"])


if __name__ == "__main__":
    train()
