from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from args import ModelArguments
from utils import logit


from sampling import (
    reparameterize_beta,
    reparameterize_bernoulli,
    reparameterize_normal,
    kl_beta,
    kl_bernoulli,
    kl_normal
)


@dataclass
class ModelOutputForLinkPrediction:
    x_hat_logits: Optional[torch.FloatTensor] = None
    edge_hat_logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_kl: Optional[torch.FloatTensor] = None
    loss_recon: Optional[torch.FloatTensor] = None


class GCNConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        cached: bool = True
    ) -> None:
        super().__init__()
        self.cached = cached
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        self.adj_norm = None

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.FloatTensor:
        """GCN layer

        Args:
            `x`: node feature matrix in shape (N, input_dim)
            `adj_mat`: adjacent matrix in shape (N, N)

        Returns:
            Node hidden units in shape (N, output_dim)
        """
        adj_norm = self._normalize_adj(adj_mat)
        x = self.linear(x)
        return adj_norm @ x

    def _normalize_adj(self, adj_mat: torch.Tensor) -> torch.Tensor:
        if self.adj_norm is None:
            # A = A + I
            adj_ = torch.eye(adj_mat.size(0)).to(adj_mat.device) + adj_mat
            # D_{ii} = (sum_{j} A_{ij})^{-1/2}
            degree_mat = adj_.sum(dim=1).pow(-0.5).diag().to_sparse()
            # degree_mat = torch.diag(adj_.sum(dim=1).pow(-0.5)).to(adj_mat.device).to_sparse()
            # normalized adjacent matrix
            adj_norm = (degree_mat @ adj_ @ degree_mat).to_sparse()
            if self.cached:
                self.adj_norm = adj_norm.requires_grad_(False)

            return adj_norm

        return self.adj_norm


class InnerProductDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features in shape (N, H)

        Returns:
            Reconstructed adj matrix logits in shape (N, N)
        """
        return x @ x.transpose(1, 0)


class DGLFRM(nn.Module):
    def __init__(self, args: ModelArguments) -> None:
        super().__init__()
        input_dim = args.num_features
        hidden_dims_enc = [input_dim] + [int(h) for h in args.hidden_dims_enc.split('_')]
        hidden_dims_dec = hidden_dims_enc[-1:] + [int(h) for h in args.hidden_dims_dec.split('_')]
        self.dropout = args.dropout
        self.temp_prior = args.temp_prior
        self.temp_post = args.temp_post
        self.K = hidden_dims_enc[-1]
        self.register_buffer("beta_a_prior", torch.FloatTensor([args.beta_a]))
        self.register_buffer("beta_b_prior", torch.FloatTensor([args.beta_b]))
        self.register_buffer("normal_mean_prior", torch.FloatTensor([0.]))
        self.register_buffer("normal_std_prior", torch.FloatTensor([1.]))

        # build model parameters
        # a and b for beta prior
        beta_a = np.log(np.exp(args.beta_a) - 1)  # inverse softplus
        beta_a = beta_a + torch.zeros(hidden_dims_enc[-1])
        beta_b = np.log(np.exp(args.beta_b) - 1)
        beta_b = beta_b + torch.zeros(hidden_dims_enc[-1])
        # shape: (K, )
        self.beta_a = nn.Parameter(beta_a)
        self.beta_b = nn.Parameter(beta_b)

        # GCN encoder
        self.encoder_feature = nn.ModuleList([
            GCNConv(dim_in, dim_out)
            for dim_in, dim_out in zip(hidden_dims_enc[:-2], hidden_dims_enc[1:-1])
        ])
        self.encoder_r_mean = GCNConv(hidden_dims_enc[-2], hidden_dims_enc[-1])
        self.encoder_r_log_std = GCNConv(hidden_dims_enc[-2], hidden_dims_enc[-1])
        self.encoder_pi_logit = GCNConv(hidden_dims_enc[-2], hidden_dims_enc[-1])

        # MLP decoder
        self.decoder_edge_hat = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.LeakyReLU()
            )
            for dim_in, dim_out in zip(hidden_dims_dec[:-2], hidden_dims_dec[1:-1])
        ])
        self.decoder_edge_hat.append(nn.Sequential(
            nn.Linear(hidden_dims_dec[-2], hidden_dims_dec[-1]),
            InnerProductDecoder(),
        ))
        self.decoder_x_hat = nn.Linear(hidden_dims_enc[-1], input_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj_mat: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None
    ) -> ModelOutputForLinkPrediction:
        """
        Args:
            `x`: node feature matrix in shape (N, D), BoW representation
            `adj_mat`: adjacent matrix in shape (N, N)
            `labels`: labels of link prediction in shape (N*N, )
            `label_mask`: mask of reconstructed loss for edges in shape (N*N, )
        """
        beta_a, beta_b, pi_logit, r_mean, r_log_std = self.encode(x, adj_mat)
        # all latent variables are in shape (N, K)
        pi_logit_prior, pi_logit_post, b, r = self.reparameterize(
            beta_a, beta_b, pi_logit, r_mean, r_log_std
        )
        if not self.training:
            b = torch.round(b)
        x_hat_logits, edge_hat_logits = self.decode(b, r)

        if labels is not None:
            loss_kl = self.compute_kl_loss(
                beta_a, beta_b,
                pi_logit_prior, pi_logit_post,
                r_mean, r_log_std
            )
            loss_recon = self.compute_recon_loss(x, x_hat_logits, edge_hat_logits, labels, label_mask)

            return ModelOutputForLinkPrediction(
                x_hat_logits,
                edge_hat_logits,
                loss_kl + loss_recon,
                loss_kl, loss_recon
            )

        return ModelOutputForLinkPrediction(x_hat_logits, edge_hat_logits)

    def encode(self, x: torch.Tensor, adj_mat: torch.Tensor) -> Tuple[torch.FloatTensor]:
        num_nodes = x.size(0)
        K = self.beta_a.size(0)

        for gcn in self.encoder_feature:
            x = F.leaky_relu(gcn(x, adj_mat))
            x = F.dropout(x, self.dropout, self.training)

        beta_a = F.softplus(self.beta_a).expand(num_nodes, K)
        beta_b = F.softplus(self.beta_b).expand(num_nodes, K)
        pi_logit = F.dropout(self.encoder_pi_logit(x, adj_mat), self.dropout, self.training)
        r_mean = F.dropout(self.encoder_r_mean(x, adj_mat), self.dropout, self.training)
        r_log_std = F.dropout(self.encoder_r_log_std(x, adj_mat), self.dropout, self.training)

        return beta_a, beta_b, pi_logit, r_mean, r_log_std

    def reparameterize(
        self,
        beta_a: torch.FloatTensor,
        beta_b: torch.FloatTensor,
        pi_logit: torch.FloatTensor,
        r_mean: torch.FloatTensor,
        r_log_std: torch.FloatTensor,
        eps: float = 1e-7
    ) -> Tuple[torch.FloatTensor]:
        v = reparameterize_beta(beta_a, beta_b)
        pi_log_prior = torch.cumsum(torch.log(v+eps), dim=1)
        pi_logit_prior = logit(torch.exp(pi_log_prior))
        pi_logit_post = pi_logit + pi_logit_prior

        self.y_sample = reparameterize_bernoulli(pi_logit_post, self.temp_post, eps)
        b = F.sigmoid(self.y_sample)

        r = reparameterize_normal(r_mean, r_log_std)

        return pi_logit_prior, pi_logit_post, b, r

    def decode(self, b: torch.FloatTensor, r: torch.FloatTensor) -> Tuple[torch.FloatTensor]:
        z = b * r
        x_hat_logits = self.decoder_x_hat(z)
        edge_hat_logits = self.decoder_edge_hat(z)

        return x_hat_logits.flatten(), edge_hat_logits.flatten()

    def compute_kl_loss(
        self,
        beta_a_post: torch.FloatTensor,
        beta_b_post: torch.FloatTensor,
        pi_logit_prior: torch.FloatTensor,
        pi_logit_post: torch.FloatTensor,
        r_mean_post: torch.FloatTensor,
        r_log_std_post: torch.FloatTensor
    ):
        num_nodes = beta_a_post.size(0)
        kl = kl_beta(self.beta_a_prior, self.beta_b_prior, beta_a_post, beta_b_post) \
            + kl_bernoulli(pi_logit_prior, pi_logit_post, self.y_sample, self.temp_prior, self.temp_post) \
            + kl_normal(self.normal_mean_prior, self.normal_std_prior, r_mean_post, r_log_std_post.exp())
        return kl / num_nodes

    def compute_recon_loss(
        self,
        x: torch.Tensor,
        x_hat_logits: torch.FloatTensor,
        edge_hat_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        num_nodes, num_features, num_edges = x.size(0), x.size(1), labels.size(0)
        # loss for reconstructed node features
        # num_neg / num_pos = #0 / #1 for BoW representation
        num_ones = x.sum()
        pos_weight_x = (num_nodes * num_features - num_ones) / num_ones
        loss_x = nn.BCEWithLogitsLoss(pos_weight=pos_weight_x).to(x.device)(
            x_hat_logits, x.to_dense().flatten()
        )

        # loss for reconstructed edges
        num_train = num_edges - label_mask.sum()
        num_pos = labels.sum()
        # (num_pos+num_neg) / (2 * num_neg)
        norm = num_edges / ((num_edges - num_pos) * 2)
        # num_neg / num_pos
        pos_weight_edge = (num_edges - num_pos) / num_pos
        loss_edge = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_edge).to(x.device)(
            edge_hat_logits, labels
        )
        if label_mask is not None:
            loss_edge.masked_fill_(label_mask, 0.0)

        return loss_x + (norm * loss_edge.sum() / num_train)
