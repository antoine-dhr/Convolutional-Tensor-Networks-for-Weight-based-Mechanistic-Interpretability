from torchvision import datasets, transforms
from torch import nn
import torch
import wandb
import torch.optim as optim

from cifar_100.cifar_100_model import CNNCIFAR100
from cifar_100.setup_cifar100 import DATA_PATH
from utils.tuning import make_sweep_train, get_best_run
from utils.train import train_model

SPLIT_SEED = 42

def main():
    # CIFAR-100 normalization stats
    train_transformation = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=5),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
            transforms.RandomErasing(p=0.1),
        ]
    )

    evaluation_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    data_path = DATA_PATH

    train = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transformation,
    )

    validation = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=evaluation_transformation,
    )

    test = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=evaluation_transformation,
    )

    total_training_samples = len(train)
    total_test_samples = len(test)
    print("Training samples:", total_training_samples)
    print("Test samples:", total_test_samples)

    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train)),
        [int(total_training_samples * 0.8), int(total_training_samples * 0.2)],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_subset = torch.utils.data.Subset(train, train_indices.indices)
    validation_subset = torch.utils.data.Subset(validation, val_indices.indices)

    sweep_train = make_sweep_train(CNNCIFAR100, train_subset, validation_subset, apply_mixup=True)

    sweep_train()

if __name__ == "__main__":
    main()
