import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from simple_parsing import ArgumentParser


@dataclass
class TrainerArguments:
    # dataset arguments
    dataset: str = "Citeseer"  # [Citeseer, Cora]
    split_name: str = "split_0.npz"  # pre-split data file
    # trainer arguments
    epochs: int = 200  # number of max epochs
    eval_step: int = 5  # evaluation frequency per training steps
    seed: Optional[int] = None  # random state
    device: str = "cuda:0"  # cuda device
    lr: float = 0.01  # initial learning rate for Adam
    weight_decay: float = 0.  # weight decay for Adam
    grad_clip: float = 10.0  # gradient clipping

    output_path: str = "output/"
    n_best_to_track: int = 3  # number of best metrics to track
    # wandb config
    experiment_name: Optional[str] = None  # wandb run name
    project: Optional[str] = None  # wandb project name
    entity: Optional[str] = None  # wandb entity

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = self.dataset
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # create directories
        self.log_dir = os.path.join(self.output_path, "logs")
        self.log_file_path = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        self.experiment_path = os.path.join(self.output_path, self.experiment_name)
        self.model_save_dir = os.path.join(self.experiment_path, "checkpoints")
        self.best_model_path = os.path.join(self.model_save_dir, "model_best.pth")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir, exist_ok=True)

        self.use_wandb = (self.project is not None)


@dataclass
class ModelArguments:
    beta_a: float = 10.  # a for beta prior
    beta_b: float = 0.1  # b for beta prior
    hidden_dims_enc: str = "64_50"  # number of hidden units of encoder layers
    hidden_dims_dec: str = "32_16"  # number of hidden units of decoder layers
    temp_prior: float = 0.5  # temperature for prior bernoulli reparameterization
    temp_post: float = 1.  # temperature for posterior bernoulli reparameterization
    dropout: float = 0.  # dropout rate (1 - keep probability)


def get_args():
    parser = ArgumentParser()
    parser.add_arguments(TrainerArguments, dest="train_args")
    parser.add_arguments(ModelArguments, dest="model_args")
    args = parser.parse_args()
    train_args: TrainerArguments = args.train_args
    model_args: ModelArguments = args.model_args

    with open(os.path.join(train_args.experiment_path, "args.json"), 'w') as file:
        json.dump({"train_args": asdict(train_args), "model_args": asdict(model_args)}, file, indent=4)

    return train_args, model_args
