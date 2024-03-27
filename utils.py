from typing import Any, Dict

import numpy as np
import torch
from torch.nn import functional as F


def logit(x: torch.FloatTensor, eps: float = 1e-7) -> torch.FloatTensor:
    return torch.log(x + eps) - torch.log(1. - x + eps)


def log_density_logistic(log_alphas, y_sample, temp):
    """
    log-density of the Logistic distribution, from
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = log_alphas + y_sample * (-temp)
    log_prob = exp_term + np.log(temp) - 2. * F.softplus(exp_term)
    return log_prob


def move_to_cuda(data: Dict[Any, torch.Tensor], device: str = "cuda") -> Dict[Any, torch.Tensor]:
    return {
        k: v.to(device) for k, v in data.items()
    }
