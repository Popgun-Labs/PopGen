import wandb
import os

from omegaconf import OmegaConf, DictConfig
from popgen import models, workers

from typing import Optional, Union


def setup_worker(name: str, cfg: Optional[Union[DictConfig, dict]] = None, exp_dir: Optional[str] = None,
                 include_wandb: bool = True, overwrite: bool = False):
    """
    :param name: unique experiment name for saving/resuming
    :param cfg: experiment config. can be `None` for existing experiment.
        otherwise a `dict` or `DictConfig` containing the configuration.
    :param exp_dir: directory to store experiment runs
    :param include_wandb: include an instance of wandb ? (required for training)
    :param overwrite: overwrite existing experiment
    :return:
    """

    # if a regular `dict` type is passed in, convert to `DictConfig` to make use of the YAML
    # serialization methods
    if type(cfg) == dict:
        cfg = DictConfig(cfg)

    # get experiment directory
    if exp_dir is None:
        exp_dir = os.environ.get('EXPERIMENT_DIR', False)
        if not exp_dir:
            raise Exception("No experiment directory defined. Set environment variable `EXPERIMENT_DIR` or "
                            "pass as kwarg `setup_worker(..., exp_dir=?)")

    # create the experiment directory if it doesn't exist
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # create the directory for this specific run
    run_dir = "{}/{}".format(exp_dir, name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # store `config.yaml` alongside the weights
    config_path = "{}/config.yaml".format(run_dir)
    if not os.path.exists(config_path):
        assert cfg is not None, "No config. supplied and no existing experiment found at {}".format(config_path)
        print("Creating new experiment.")

        # generate a unique id for this run
        run_id = wandb.util.generate_id()
        OmegaConf.set_struct(cfg, False)
        cfg['run_id'] = run_id

        OmegaConf.save(cfg, config_path)
    else:
        if not overwrite:
            print("Loading existing experiment at {}".format(config_path))
            print("If you would like to overwrite, please set overwrite=True.")
            cfg = OmegaConf.load(config_path)
        else:
            print("Overwriting existing configuration at {}".format(config_path))
            OmegaConf.save(cfg, config_path)

    # print the configuration
    print(cfg.pretty())

    # initialise model
    model_class = getattr(models, cfg['model_class'])
    model = model_class(**cfg['model'])

    # setup visualisation
    run = None
    if include_wandb and 'wandb' in cfg:
        run = wandb.init(
            name=name,
            config=cfg,
            id=cfg['run_id'],
            resume='allow',
            **cfg['wandb']
        )
        run.watch(model)

    # initialise the worker
    worker_class = getattr(workers, cfg['worker_class'])
    worker = worker_class(name, model, run_dir, run, **cfg['worker'])

    return worker, cfg
