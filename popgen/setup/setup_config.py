import os
from typing import Optional, Union

import wandb
from omegaconf import DictConfig, OmegaConf


def setup_config(
    name: str,
    cfg: Optional[Union[DictConfig, dict]] = None,
    exp_dir: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> DictConfig:
    """
    Create an experiment with the specific name and settings
    OR create a new experiment if it doesn't exist
    OR overwrite the configuration of an existing experiment

    :param name: unique experiment name for saving/resuming
    :param cfg: experiment config. can be `None` for existing experiment.
        otherwise a `dict` or `DictConfig` containing the configuration.
    :param exp_dir: directory to store experiment runs
    :param overwrite: overwrite existing experiment
    :param verbose: print out the loaded config
    :return:
    """
    # if a regular `dict` type is passed in, convert to `DictConfig` to make use of the YAML
    # serialization methods
    if type(cfg) == dict:
        cfg = DictConfig(cfg)

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
        cfg["run_id"] = run_id

        OmegaConf.save(cfg, config_path)
    # Note: when overwriting parameters we will preserve the same `run_id`. This allows to resume
    # training with tweaked parameters (e.g lower learning rate)
    else:
        existing_cfg = OmegaConf.load(config_path)
        run_id = existing_cfg["run_id"]

        if not overwrite:
            print("Loading existing experiment at {}".format(config_path))
            print("If you would like to overwrite, please set overwrite=True.")
            cfg = existing_cfg
        else:
            print("Overwriting existing configuration at {}".format(config_path))
            OmegaConf.set_struct(cfg, False)
            cfg["run_id"] = run_id
            OmegaConf.save(cfg, config_path)

    # print the configuration
    if verbose:
        print(OmegaConf.to_yaml(cfg))

    return cfg
