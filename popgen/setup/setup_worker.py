import os
import wandb

from omegaconf import DictConfig
from typing import Optional, Union

from popgen.setup.setup_config import setup_config
from popgen.setup.utils import import_pkg


def setup_worker(
    name: str,
    cfg: Optional[Union[DictConfig, dict]] = None,
    exp_dir: Optional[str] = None,
    include_wandb: bool = True,
    overwrite: bool = False,
    module: Optional[str] = None,
):
    """
    :param name: unique experiment name for saving/resuming
    :param cfg: experiment config. can be `None` for existing experiment.
        otherwise a `dict` or `DictConfig` containing the configuration.
    :param exp_dir: directory to store experiment runs
    :param include_wandb: include an instance of wandb ? (required for training)
    :param overwrite: overwrite existing experiment
    :param module: a python module containing `workers`, `models` and `datasets`
    :return:
    """
    if module is not None:
        workers = import_pkg(module, "workers")
        models = import_pkg(module, "models")
    else:
        from popgen import workers, models

        print("Warning: No module supplied in `setup_loaders`. Defaulting to `popgen.workers` and `popgen.models`. ")

    # get experiment directory
    if exp_dir is None:
        exp_dir = os.environ.get("EXPERIMENT_DIR", False)
        if not exp_dir:
            raise Exception(
                "No experiment directory defined. Set environment variable `EXPERIMENT_DIR` or "
                "pass as kwarg `setup_worker(..., exp_dir=?)"
            )

    # setup the directory
    cfg = setup_config(name, cfg, exp_dir, overwrite)

    # initialise model
    model_class = getattr(models, cfg["model_class"])
    model = model_class(**cfg["model"])

    # setup visualisation
    run = None
    if include_wandb and "wandb" in cfg:
        run = wandb.init(name=name, config=cfg, id=cfg["run_id"], resume="allow", **cfg["wandb"])
        run.watch(model)

    # initialise the worker
    run_dir = "{}/{}".format(exp_dir, name)
    worker_class = getattr(workers, cfg["worker_class"])
    worker = worker_class(name, model, run_dir, run, **cfg["worker"])

    return worker, cfg
