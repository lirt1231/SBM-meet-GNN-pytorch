import numpy as np
import torch
from torch.distributions import Normal, Beta, kl_divergence

from utils import log_density_logistic


def reparameterize_beta(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
) -> torch.FloatTensor:
    return Beta(a, b).rsample()


def reparameterize_bernoulli(
    log_pi: torch.FloatTensor,
    temperature: float = 1.,
    eps: float = 1e-7
) -> torch.FloatTensor:
    """Reparameterization of Bernoulli distribution with Binary Concrete distritbution."""
    u = torch.from_numpy(
        np.random.uniform(eps, 1., log_pi.size()).astype(np.float32)
    ).to(log_pi.device)
    logistic = torch.log(u) - torch.log(1. - u)

    return (log_pi + logistic) / temperature


def reparameterize_normal(
    mean: torch.FloatTensor,
    log_std: torch.FloatTensor
) -> torch.FloatTensor:
    """Reparameterization of Normal distribution."""
    return Normal(mean, log_std.exp()).rsample()


def kl_beta(
    beta_a_prior: torch.FloatTensor, beta_b_prior: torch.FloatTensor,
    beta_a: torch.FloatTensor, beta_b: torch.FloatTensor
) -> torch.Tensor:
    return kl_divergence(
        Beta(beta_a_prior, beta_b_prior),
        Beta(beta_a, beta_b)
    ).sum(dim=1).mean()


def kl_bernoulli(
    pi_logit_prior: torch.FloatTensor,
    pi_logit_post: torch.FloatTensor,
    y_sample: torch.FloatTensor,
    temp_prior: float = 0.5,
    temp_post: float = 1.,
) -> torch.Tensor:
    log_prior = log_density_logistic(pi_logit_prior, y_sample, temp_prior)
    log_posterior = log_density_logistic(pi_logit_post, y_sample, temp_post)
    kl = log_posterior - log_prior

    return kl.sum(dim=1).mean()


def kl_normal(
    r_mean_prior: torch.FloatTensor, r_std_prior: torch.FloatTensor,
    r_mean_post: torch.FloatTensor, r_std_post: torch.FloatTensor
) -> torch.Tensor:
    return kl_divergence(
        Normal(r_mean_prior, r_std_prior),
        Normal(r_mean_post, r_std_post)
    ).sum(dim=1).mean()
