import torch
import torch.nn as nn
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

from torch.utils.data import DataLoader


class AbstractWorker(ABC):
    def __init__(self, exp_name, model, run_dir, wandb=None, epoch_save_freq=50, log_interval=50, *args, **kwargs):
        """
        All workers should inherit from this class. It will:
        1. provide common utilities for saving, loading and device management.
        2. enforce a common interface between workers. For example, all must implement `.train()` and `evaluate.()`

        :param exp_name: name of experiment
        :param model: instance of a PyTorch model (nn.Module)
        :param run_dir: directory to save configurations, weights, artifacts for this experiment.
        :param wandb: the session that is returned from wandb.init(...).
        :param epoch_save_freq: how often to save intermediate checkpoints
        :param log_interval: how many gradient updates before logging to visdom
        :param args:
        :param kwargs:
        """

        self.exp_name = exp_name
        self.wandb = wandb
        self.run_dir = run_dir
        self.upload_checkpoints = kwargs.get("upload_checkpoints", False)

        # keep a list of all stateful objects, to include in the experiment checkpoints
        # includes the model by default
        # see: `register_state()
        self.stateful_objects = {
            'model': model
        }

        # track the loss
        self.lowest_loss = float('inf')
        self.epoch_save_freq = epoch_save_freq
        self.log_interval = log_interval

        # cache values until its time to plot them
        self._counters = {
            'train': 0,
            'test': 0
        }
        self._metric_cache = {
            'train': {},
            'test': {}
        }

    @abstractmethod
    def train(self, loader):
        pass

    @abstractmethod
    def evaluate(self, loader):
        pass

    def run(self, train_loader: DataLoader, test_loader: DataLoader, nb_epoch: int):
        """
        Run an experiment for the specified number of epoch.
        :param train_loader:
        :param test_loader:
        :param nb_epoch:
        :return:
        """
        for epoch in range(nb_epoch):
            # reset numpy random seed
            np.random.seed()

            # wandb train and test sets
            self.train(train_loader)
            with torch.no_grad():
                loss_score, *_ = self.evaluate(test_loader)

            # save `best`
            if loss_score < self.lowest_loss:
                print("New lowest test loss {}".format(loss_score))
                self.save(checkpoint_id='best')
                self.lowest_loss = loss_score
                if self.wandb is not None:
                    self.wandb.summary["lowest_loss"] = loss_score

            # save every `x` epoch
            if epoch % self.epoch_save_freq == 0:
                self.save(checkpoint_id="{}".format(epoch))

            # overwrite latest weights
            self.save(checkpoint_id='latest')

    def cuda(self, device_id: int = 0):
        """
        Move the model and all optimisers to the GPU.
        :return:
        """
        assert torch.cuda.is_available(), "CUDA support not found!"
        for key, obj in self.stateful_objects.items():
            if isinstance(obj, AbstractWorker):
                continue
            if hasattr(obj, "cuda"):
                self.stateful_objects[key].cuda(device_id)

    def register_state(self, obj: Any, name: str):
        """
        Register an object to be included in the experiment checkpoint.
        Any object implementing `.load_state_dict()` and `.state_dict()` is valid. For example:
            - PyTorch optimiser, LR Scheduler or Model
            - Any custom class implementing those methods (e.g. a specific worker with extra state)

        :param obj: obj
        :param name: unique name for what is being registered
        """
        assert name not in self.stateful_objects, "Duplicate key in state list for '{}'".format(name)
        assert hasattr(obj, 'load_state_dict'), "Object must implement `load_state_dict` to be included in checkpoint."
        assert hasattr(obj, 'state_dict'), "Object must implement `state_dict` to be included in checkpoint."

        self.stateful_objects[name] = obj

    def save(self, checkpoint_id: str = "latest"):
        """
        Save state of all tracked modules.
        :param checkpoint_id:
        :return:
        """
        state_dict = {}
        for key, obj in self.stateful_objects.items():
            state_dict[key] = obj.state_dict()

        checkpoint_path = "{}/checkpoint_{}.pt".format(self.run_dir, checkpoint_id)
        print("Saving checkpoint {}".format(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

        # upload and overwrite the checkpoints to wandb if `upload_checkpoints` enabled in worker config
        if self.wandb is not None and self.upload_checkpoints and checkpoint_id in ["latest", "best"]:
            self.wandb.save(
                glob_str=checkpoint_path,
                base_path=str(Path(self.run_dir).parent),
                policy="live"
            )

    def load(self, checkpoint_id: str = "best", strict: bool = True):
        """
        Load state from existing checkpoint.
        :param checkpoint_id:
        :param strict: whether to use strict loading for the model weights (see PyTorch nn.Module `load_state_dict)
        :return:
        """

        checkpoint_path = "{}/checkpoint_{}.pt".format(self.run_dir, checkpoint_id)

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found {}".format(checkpoint_path))
            return

        print("Loading checkpoint {}".format(checkpoint_path))

        state_dict = torch.load(checkpoint_path)
        for key, state in state_dict.items():
            assert key in self.stateful_objects, "Invalid key `{}` in saved checkpoint.".format(key)
            obj = self.stateful_objects[key]
            if isinstance(obj, nn.Module):
                obj.load_state_dict(state, strict)
            else:
                obj.load_state_dict(state)

    @staticmethod
    def unwrap_value(v):
        if torch.is_tensor(v):
            return v.item()
        elif type(v) == np.ndarray:
            return v.item()
        else:
            return v

    def _plot_loss(self, metrics: dict, train=True):
        """
        Plot metrics to weights and biases.

        Keeps a moving average of values and pushes them every `log_interval`.

        :param metrics: dictionary of things to track
        :param train: ?
        """
        subset = "train" if train else "test"

        for k, v in metrics.items():
            full_key = "{} {}".format(k, subset)
            if full_key not in self._metric_cache[subset]:
                self._metric_cache[subset][full_key] = []
            v_raw = AbstractWorker.unwrap_value(v)
            self._metric_cache[subset][full_key].append(v_raw)

        # increment the counter
        self._counters[subset] += 1

        # empty cache and plot!
        if self._counters[subset] % self.log_interval == 0:
            avg = {}
            for k, v in self._metric_cache[subset].items():
                avg[k] = np.mean(v)
                self._metric_cache[subset][k] = []

            self.wandb.log(avg)
            self._counters[subset] = 0
