import torch
import wandb

import logger
from args import get_args
from dataset import GraphDataset
from model import DGLFRM
from trainer import Trainer


if __name__ == '__main__':
    train_args, models_args = get_args()
    logger.create_logger(train_args)
    dataset = GraphDataset(train_args.dataset, train_args.split_name)
    models_args.num_features = dataset.num_features
    model = DGLFRM(models_args)
    trainer = Trainer(train_args, model, dataset)
    print(model)

    # train model
    trainer.train_loop()
    # load state dict and test model
    model.load_state_dict(torch.load(train_args.best_model_path, map_location=train_args.device))
    roc_score, ap_score, auc_pr = trainer.eval("test")
    metric_dict = {
        "test/roc": roc_score,
        "test/ap": ap_score,
        "test/auc_pr": auc_pr
    }
    logger.log(f"[test] {metric_dict}")

    if train_args.use_wandb:
        for k, v in metric_dict.items():
            wandb.summary[k] = v
        wandb.finish()
