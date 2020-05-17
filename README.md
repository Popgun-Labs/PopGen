# PopGen

PopGen is a generative modelling research toolkit for PyTorch, with an emphasis on likelihood-based models
and representation learning. It aims to provide high quality reference implementations, evaluation code
and reusable model components.

PopGen is still a young project. More models and code examples will be added in coming months.

## Example Experiment

The included VAE examples demonstrate how flexible posterior and prior distributions can
improve over a gaussian baseline. The `vae` and `vamp` architectures follow the settings
of the L=1 VAE described in [VAE with a VampPrior](https://arxiv.org/abs/1705.07120). The `vae_vamp_hsnf`
model also introduces K=4 [Sylvester Normalizing Flows](https://arxiv.org/abs/1803.05649) to the
posterior distribution.

Each model is trained for 1M steps on dynamically binarized MNIST. The checkpoint with the lowest
test loss is retained. Marginal likelihoods are estimated using the [IWAE](https://arxiv.org/abs/1509.00519) bound
and 5000 samples.

| Name | Posterior | Prior | log p(x) |
| --- | --- | --- | --- |
| `vae` | Diagonal Gaussian | Standard Gaussian | -84.04 |
| `vae_vamp` | Diagonal Gaussian | VAMP Prior | -81.96 |
| `vae_vamp_hsnf` | Sylvester Flows | VAMP Prior | -80.69 |


PopGen uses [Weights and Biases](https://www.wandb.com/) for visualisation.
See the training [plots](https://app.wandb.ai/angusturner/vae_experiments?workspace=user-angusturner) for this example.


## Implemented Papers and Modules

| Concept | Implementation | Associated Paper(s) |
| --- | --- | --- |
| Causal Convolution | `popgen.nn.causal_conv` | [WaveNet](https://arxiv.org/abs/1609.03499), [Fast Wavenet](https://arxiv.org/abs/1611.09482) |
| Sylvester Flows | `popgen.nn.flows.hsnf` | [Sylvester Normalizing Flows](https://arxiv.org/abs/1803.05649) |
| VAMP Prior | `popgen.nn.vamp_prior` | [VAE with a VampPrior](https://arxiv.org/abs/1705.07120) |
| Vector Quantization | `popgen.nn.vqvae` | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) |
| Constrained Optimisation for Beta-VAE | `popgen.optim.geco` | [Taming VAEs](https://arxiv.org/abs/1810.00597) |
| Contrastive Predictive Coding | `popgen.nn.cpc` | [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) |

## Requirements

- Python >= 3.6
- PyTorch >=1.4 (may work with older versions)
- GPU with CUDA compatibility
- (recommended) Linux

## Installation

### As a module, to use components in other projects:

Prefer this option if you want to use PopGen components as part of another project.

```shell script
git clone https://github.com/Popgun-Labs/popgen.git
pip install -e popgen
```

### As an experiment runner / research framework:

Prefer this option if you want to use the PopGen framework to run experiments or
the included notebooks.

```shell script
git clone https://github.com/Popgun-Labs/popgen.git
cd popgen
pip install -r requirements.txt
```

## Experiment Framework

Experiments in PopGen are organised around three core concepts.

### Models

- Any PyTorch model (inherits from `nn.Module`)
- Should not contain optimisation related code.
- Must be included `models/__init__.py` to be dynamically loaded by the experiment runner.
- Refer to `models/vae.py` for an example.

### Workers

- Responsible for training and evaluating models
- Must implement `.train` and `.evaluate`
- Should inherit from the `AbstractWorker` class, which will:
    1. provide utility methods for saving and loading experiment checkpoints.
    2. enforce a common training interface
- Must be included in `workers/__init__.py` to be dynamically loaded by the experiment runner.
- Refer to `workers/vae_worker.py` for an example.

### Datasets

- Must implement `__len__` and `__getitem__`
- Will be wrapped with the PyTorch `DataLoader` class for batching and collation
- Must be included in `datasets/__init__.py` to be dynamically loaded by the experiment runner.
- Refer to `datasets/binary_mnist.py` for an example.

Note that there is not a strict 1:1 relationship between workers and models, models and datasets etc.
It is up to the developer to ensure a compatible API between these various components.

## Configuring an Experiment

The following environment variables should be set:
- `EXPERIMENT_DIR`. An absolute path to the desired model output location.
- `WANDB_API_KEY`. An API key for [Weights and Biases](https://www.wandb.com/)

Experiment configuration is managed with [Hydra](https://hydra.cc/).
The main entry point is `config/config.yaml`:
```yaml
name: null
model: null
dataset: null
worker: null
nb_epoch: 1667
wandb:
  project: vae_experiments
defaults:
  - loader: basic
```

The `null` entries signify items that should be set on the CLI.
- `name` is the experiment name. can be any string.
- `model` selects a model configuration from the `config/model` directory.
- `dataset` selects a dataset configuration from `config/dataset`
- `worker` selects a worker configuration from `config/worker`

Defaults can be set, as per the `loader` entry.

For example,
```shell script
python train.py name=vae model=vae dataset=binary_mnist worker=vae_worker
```
yields a config like:
```yaml
dataset:
  both:
    data_dir: /home/angusturner/data/
    dynamic: true
  test:
    train: false
  train:
    train: true
dataset_class: BinaryMNIST
experiment_dir: /home/angusturner/experiments/
loader:
  both:
    batch_size: 100
    num_workers: 4
    pin_memory: true
    shuffle: true
  test:
    drop_last: false
  train:
    drop_last: true
model:
  decoder: {}
  encoder:
    deterministic: false
  hidden_dim: 300
  input_dim: 784
  latent_dim: 40
  posterior_flow: null
  prior: null
model_class: VAE
name: vae
nb_epoch: 1667
overwrite: false
run_id: 212lzoyk
wandb:
  project: vae_experiments
worker:
  annealing_temp: 0.0001
  epoch_save_freq: 50
  log_interval: 50
  max_beta: 1.0
  optim_class: Adam
  optim_settings:
    lr: 0.0002
    weight_decay: 0.0
worker_class: VAE_Worker
```

Any entry of this config can be overwritten on the CLI. For example, overwriting the batch size:
```shell script
python train.py ... loader.both.batch_size=16
```

For more information on how Hydra works refer to the [Hydra docs](https://hydra.cc/docs/intro).

## Contributors

```text
Angus Turner - angus@wearepopgun.com
Liang Zhang - liang@wearepopgun.com
Adam Hibble - adam@wearepopgun.com
Rhys Perren - rhys@wearepopgun.com
```

## Other References

PopGen would not exist without these excellent open-source resources:
- [Pytorch](https://pytorch.org/)
- Eric Jang's [Blog on Normalising Flows](https://blog.evjang.com/2018/01/nf1.html)
- Jakub Tomczak's [VAE research code](https://github.com/jmtomczak/vae_vampprior)
