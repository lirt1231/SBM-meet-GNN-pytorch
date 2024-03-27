import bisect
import os
import shutil
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

import logger
from args import TrainerArguments
from logger import MetricTracker, BestMetricTracker
from dataset import GraphDataset
from model import DGLFRM, ModelOutputForLinkPrediction
from utils import move_to_cuda


class Trainer:
    def __init__(
        self,
        train_args: TrainerArguments,
        model: DGLFRM,
        dataset: GraphDataset
    ) -> None:
        self.args = train_args
        self.model = model.to(train_args.device)
        self.dataset = dataset

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_args.lr,
            weight_decay=train_args.weight_decay
        )
        # number of models with best evaluation metrics to save
        self.n_best = train_args.n_best_to_track
        self.best_metric_list = deque()

    def train_loop(self) -> None:
        roc_tracker = MetricTracker("roc", "valid")
        ap_tracker = MetricTracker("ap", "valid")
        auc_pr_tracker = MetricTracker("auc_pr", "valid")
        best_roc_tracker = BestMetricTracker("best_roc", "valid")

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)

            if (epoch+1) % self.args.eval_step == 0:
                roc_score, ap_score, auc_pr = self.eval("valid")

                roc_tracker.update(roc_score)
                ap_tracker.update(ap_score)
                auc_pr_tracker.update(auc_pr)
                best_roc_tracker.update(roc_score, epoch)
                logger.log_metric(
                    roc_tracker, ap_tracker, auc_pr_tracker,
                    best_roc_tracker, step=epoch
                )

                self._maybe_save(best_roc_tracker.metric)

    def train_epoch(self, epoch: int) -> None:
        model = self.model.train()

        loss_tracker = MetricTracker("loss", "train")
        loss_kl_tracker = MetricTracker("loss_kl", "train")
        loss_recon_tracker = MetricTracker("loss_recon", "train")

        data = self.dataset.collate(self.dataset[0])
        data = move_to_cuda(data, self.args.device)
        output = model(**data)

        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_tracker.update(loss.item())
        loss_kl_tracker.update(output.loss_kl.item())
        loss_recon_tracker.update(output.loss_recon.item())
        logger.log(f"Epoch {epoch}")
        logger.log_metric(loss_tracker, loss_kl_tracker, loss_recon_tracker, step=epoch)

    @torch.no_grad()
    def eval(self, data_type: str = "valid") -> None:
        model = self.model.eval()

        data = self.dataset[0]
        x, adj_mat = data["node_features"], data["adj_train"]
        # construct label and mask
        if data_type == "valid":
            edges_true = data["edges_val"], data["edges_val_neg"]
        elif data_type == "test":
            edges_true = data["edges_test"], data["edges_test_neg"]
        else:
            raise ValueError(f"unknown data type {data_type}")
        data = move_to_cuda({"x": x, "adj_mat": adj_mat}, self.args.device)
        output = model(**data)

        return self.compute_metrics(output, edges_true[0], edges_true[1])

    def compute_metrics(
        self,
        output: ModelOutputForLinkPrediction,
        edges_pos: np.ndarray,
        edges_neg: np.ndarray
    ) -> Tuple[float]:
        """Compute evaluation metrics for the link prediction task.

        Args:
            `output`: model output
            `edges_pos`: positive edges with shape (E1, 2)
            `edges_neg`: negative edges with shape (E2, 2)

        Returns:
            roc_score, ap_score and auc_pr
        """
        labels = np.hstack([np.ones(len(edges_pos)), np.zeros(len(edges_neg))])

        pred_prob = []
        logits = output.edge_hat_logits.detach().cpu()
        for e in np.vstack((edges_pos, edges_neg)):
            pred_prob.append(F.sigmoid(logits[e[0]*self.dataset.num_nodes + e[1]]))
        pred_prob = np.array(pred_prob)

        roc_score = roc_auc_score(labels, pred_prob)
        ap_score = average_precision_score(labels, pred_prob)
        precision, recall, _ = precision_recall_curve(labels, pred_prob)
        auc_pr = auc(recall, precision)

        return roc_score, ap_score, auc_pr

    def _maybe_save(self, best_metric: BestMetricTracker.BestMetric) -> None:
        epoch, metric = best_metric.epoch, best_metric.value
        if best_metric in self.best_metric_list:
            return

        bisect.insort(self.best_metric_list, best_metric)
        # remove the model state with the worst metric
        path_template = os.path.join(self.args.model_save_dir, "roc{m}_epoch{e}.pth")
        if len(self.best_metric_list) > self.n_best:
            metric, epoch = self.best_metric_list.popleft().get()
            path = path_template.format(m=metric, e=epoch)
            os.remove(path)
        # save models in the list
        for metric, epoch in map(lambda x: x.get(), self.best_metric_list):
            path = path_template.format(m=metric, e=epoch)
            if not os.path.exists(path):
                torch.save(self.model.state_dict(), path)
        # create symbolic link to the best model
        metric, epoch = self.best_metric_list[-1].get()
        shutil.copy(path_template.format(m=metric, e=epoch), self.args.best_model_path)

    def _save(self, path: str) -> None:
        if not os.path.exists(path):
            torch.save(self.model.state_dict(), path)
