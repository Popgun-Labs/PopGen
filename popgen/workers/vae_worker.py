import torch
import torch.nn.functional as F
import torch.optim
import wandb
import numpy as np

from tqdm.auto import tqdm

from popgen.utils import sigmoid_annealing
from popgen.workers.abstract_worker import AbstractWorker
from popgen.distributions import reparameterize


class VAE_Worker(AbstractWorker):
    def __init__(
        self, exp_name, model, run_dir, wandb, optim_class, optim_settings, annealing_temp, max_beta, *args, **kwargs
    ):
        super(VAE_Worker, self).__init__(exp_name, model, run_dir, wandb, *args, **kwargs)

        self.annealing_temp = annealing_temp
        self.max_beta = max_beta
        self.model = model

        # setup the optimiser
        self.params = [p for p in model.parameters() if p.requires_grad]
        optim_class = getattr(torch.optim, optim_class)
        self.optim = optim_class(self.params, **optim_settings)

        # register the optimiser and worker state, to be saved in the checkpoints
        self.register_state(self, "worker")
        self.register_state(self.optim, "optim")

        # put everything on GPU if available
        if torch.cuda.is_available():
            self.cuda()

        # track the number of gradient updates
        self.nb_iters = 0

        # load existing checkpoint
        self.load(checkpoint_id="best")

    def _get_beta(self):
        beta = sigmoid_annealing(self.nb_iters, self.annealing_temp)
        beta *= self.max_beta
        return beta

    def _step(self):
        self.optim.step()
        self.nb_iters += 1

    def main(self, loader, train=True):
        """
        Implement the model optimisation logic.
        :param loader: torch.utils.data.DataLoader
        :param train: train on this data?
        :return:
        """
        elbos = []
        for i, (x, y) in enumerate(tqdm(loader)):
            if train:
                self.optim.zero_grad()

            # put features on GPU
            x = x.cuda().view(-1, 784)

            # forward pass
            logits, nll, log_p_z, log_q_z, ln_det = self.model(x)

            # compute loss
            beta = self._get_beta()
            kl = log_q_z - log_p_z
            loss = nll + beta * kl

            # track the elbo
            elbo = -nll - kl
            elbos.append(-elbo.item())

            # optimise
            if train:
                loss.backward()
                self._step()

            # plot loss
            if i % self.log_interval == 0:
                self._plot_loss(
                    {
                        "loss": loss,
                        "ELBO": elbo.item(),
                        "NLL": nll.item(),
                        "KL": kl.item(),
                        "log q(z|x)": log_q_z,
                        "log p(z)": log_p_z,
                        "ln det": ln_det,
                        "Beta": beta,
                    },
                    train=train,
                )

            # plot samples
            if i % 500 == 0:
                x_hat = torch.sigmoid(logits).round()
                self._plot_samples(x, x_hat, train=train)

        return (np.mean(elbos),)

    def train(self, loader):
        self.model.train()
        return self.main(loader, train=True)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        return self.main(loader, train=False)

    @torch.no_grad()
    def estimate_likelihood(self, loader, nb_samples=5000, mb=1000):
        """
        Computes a `k` sample importance weighted estimate of the marginal likelihood p(x)
        Note. Optimising this objective directly leads to IWAE
        Reference: https://arxiv.org/pdf/1509.00519.pdf
        :param nb_samples: the number of posterior samples to use
        :param param mb: the decoding mini-batch size (should evenly divide `nb_samples`)
        """
        self.model.eval()

        assert nb_samples % mb == 0, "`mb` must evenly divide `nb_samples`"

        REs, KLs, ELBOs = [], [], []
        for i, (x, y) in enumerate(tqdm(loader)):
            # cast features to variables
            x = x.view(x.size(0), -1)
            x, y = x.cuda(), y.cuda()

            # get parameters of base posterior
            mu, logvar, h = self.model.encoder(x)

            # iterate over batch
            for j, (mu_j, logvar_j, h_j) in enumerate(zip(mu, logvar, h)):
                # draw `nb_samples` from the base posterior (batch, nb_samples)
                z_0 = reparameterize(mu_j.repeat(nb_samples, 1), logvar_j.repeat(nb_samples, 1))

                # compute reconstructions in batches
                z_0s = torch.chunk(z_0, nb_samples // mb, dim=0)
                RE_j, KL_j = [], []
                for z_0 in z_0s:
                    # pass through flows
                    z_K = z_0
                    ln_det = torch.zeros(mb, device=z_0.device)
                    if self.model.posterior_flow is not None:
                        z_K, ln_det = self.model.posterior_flow(z_K, h_j.repeat(mb, 1))
                        ln_det = ln_det.sum(-1)  # (batch)

                    # compute KL
                    log_p_z = self.model.log_p_z(z_K, reduce=False, cache_params=True)
                    log_q_z = (
                        self.model.log_q_z0(z_0, mu_j.repeat(mb, 1), logvar_j.repeat(mb, 1), reduce=False) - ln_det
                    )
                    KL = log_q_z - log_p_z
                    KL_j.append(KL.cpu())

                    # decode
                    x_sample_j = self.model.decoder(z_K)

                    # compute likelihood
                    RE = F.binary_cross_entropy_with_logits(x_sample_j, x[j : j + 1].repeat(mb, 1), reduction="none")
                    RE = torch.sum(RE, dim=-1, keepdim=False)
                    RE_j.append(RE.cpu())

                # concatenate samples for this data point (nb_samples)
                RE_j = torch.cat(RE_j, dim=0)
                KL_j = torch.cat(KL_j, dim=0)
                ELBO_j = -(RE_j + KL_j)

                # compute the log of the average probability
                ELBO_j = torch.logsumexp(ELBO_j, dim=0, keepdim=False)
                ELBO_j = ELBO_j - np.log(nb_samples)
                ELBOs.append(ELBO_j)

        # concatenate and average over dataset
        ELBOs = torch.stack(ELBOs, dim=0)
        ELBO = torch.mean(ELBOs).item()

        return ELBO

    def _plot_loss(self, metrics: dict, train=True):
        """
        :param metrics: dictionary of things to track
        :param train: ?
        """
        suffix = "Train" if train else "Test"
        metrics_ = {}
        for k, v in metrics.items():
            metrics_["{} {}".format(k, suffix)] = v
        self.wandb.log(metrics_)

    def _plot_samples(self, x, x_hat, train=True, nb_examples=4):
        """
        :param x: batch of ground truth torch.tensor (batch, 784)
        :param x_hat: batch of reconstructions (batch, 784)
        :param nb_examples, number of examples to plot...
        :param train
        """
        prefix = "Train" if train else "Test"

        # convert to wandb images
        orig, rec = [], []
        for i in range(nb_examples):
            x_np = x[i].detach().cpu().view(28, 28).numpy()
            x_hat_np = x_hat[i].detach().cpu().view(28, 28).numpy()
            orig.append(wandb.Image(x_np))
            rec.append(wandb.Image(x_hat_np))

        # plot
        self.wandb.log({"{} Originals".format(prefix): orig, "{} Reconstructions".format(prefix): rec})

    def state_dict(self):
        return {"nb_iters": self.nb_iters}

    def load_state_dict(self, state):
        self.nb_iters = state["nb_iters"]
