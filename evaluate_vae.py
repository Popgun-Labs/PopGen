import hydra
from omegaconf import DictConfig

from popgen.setup import setup_worker, setup_loaders


@hydra.main()
def evaluate_vae(cfg: DictConfig) -> None:
    # get the experiment name
    name = cfg.get("name", False)
    if not name:
        raise Exception("Must specify experiment name on CLI. e.g. `python evaluate_vae.py name=vae`")

    # setup the worker
    worker, cfg = setup_worker(name, include_wandb=False, overwrite=False)

    # setup data loaders
    _train_loader, test_loader = setup_loaders(
        dataset_class=cfg["dataset_class"], data_opts=cfg["dataset"], loader_opts=cfg["loader"]
    )

    # train
    elbo = worker.estimate_likelihood(test_loader)
    print("Evidence Lower Bound: {}".format(elbo))


if __name__ == "__main__":
    evaluate_vae()
