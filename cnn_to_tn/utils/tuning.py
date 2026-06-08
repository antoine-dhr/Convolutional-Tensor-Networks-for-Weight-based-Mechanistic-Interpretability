import wandb
import torch.optim as optim
from torch import nn
from utils.train import train_model

def make_sweep_train(model_cls, train_subset, val_subset, apply_mixup=True):
    def sweep_train():
        run = wandb.init()
        config = wandb.config

        model = model_cls(
            config
        )

        optimizer_arg = None 

        if config.optimizer == "adam":
            optimizer_arg = optim.Adam
        elif config.optimizer == "adamw":
            optimizer_arg = optim.AdamW
        elif config.optimizer == "sgd":
            optimizer_arg = optim.SGD
        else:
            raise ValueError(f"Optimizer arg {config.optimizer} not supported")

        train_model(
            model=model,
            train_dataset=train_subset,
            test_dataset=val_subset,
            config=config,
            optimizer_arg=optimizer_arg,
            criterion_arg=nn.CrossEntropyLoss,
            tuning_flag=True,
            apply_mixup=apply_mixup
        )

        run.finish()

    return sweep_train
