from torchvision import datasets, transforms
from torch import nn
import torch
import wandb
import torch.optim as optim

from SVHN.svhn_model import CNNSVHN
from SVHN.setup_svhn import DATA_PATH
from utils.tuning import make_sweep_train, get_best_run
from utils.train import train_model

SPLIT_SEED = 42

def main():
    # SVHN normalization stats (computed from training set)
    train_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.1980, 0.2010, 0.1970),
            ),
        ]
    )

    evaluation_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.1980, 0.2010, 0.1970),
            ),
        ]
    )

    data_path = DATA_PATH

    train = datasets.SVHN(
        root=data_path,
        split="train",
        download=True,
        transform=train_transformation,
    )

    extra = datasets.SVHN(
        root=data_path,
        split="extra",
        download=True,
        transform=train_transformation,
    )

    validation = datasets.SVHN(
        root=data_path,
        split="train",
        download=True,
        transform=evaluation_transformation,
    )

    test = datasets.SVHN(
        root=data_path,
        split="test",
        download=True,
        transform=evaluation_transformation,
    )

    total_training_samples = len(train)
    total_extra_samples = len(extra)
    total_test_samples = len(test)
    print("Training samples:", total_training_samples)
    print("Extra samples:", total_extra_samples)
    print("Test samples:", total_test_samples)

    n_train = int(total_training_samples * 0.8)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train)),
        [n_train, total_training_samples - n_train],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_main_subset = torch.utils.data.Subset(train, train_indices.indices)
    train_subset = torch.utils.data.ConcatDataset([train_main_subset, extra])
    validation_subset = torch.utils.data.Subset(validation, val_indices.indices)

    sweep_train = make_sweep_train(CNNSVHN, train_subset, validation_subset, apply_mixup=False)

    sweep_train()

if __name__ == "__main__":
    main()
