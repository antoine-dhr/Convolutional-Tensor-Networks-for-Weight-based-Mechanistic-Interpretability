from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch import nn
import torch
import matplotlib.pyplot as plt
import wandb
import torch.optim as optim

from cifar_10.cifar_10_model import CNNCIFAR10
from cifar_10.setup_cifar10 import DATA_PATH
from utils.tuning import make_sweep_train, get_best_run
from utils.train import train_model

SPLIT_SEED = 42

def main():
    train_transformation = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
        transforms.RandomErasing(p=0.25)
    ]
    )

    evaluation_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ]
    )

    data_path = DATA_PATH

    train = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=train_transformation
    )

    validation = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=evaluation_transformation
    )

    test = datasets.CIFAR10(
        root=data_path, 
        train=False, 
        download=False, 
        transform=evaluation_transformation
    )
        

    total_training_samples = len(train)
    total_test_samples = len(test)
    print("Training samples: ", total_training_samples)
    print("Test samples: ", total_test_samples)

    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train)),
        [int(total_training_samples * 0.8), int(total_training_samples * 0.2)],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_subset = torch.utils.data.Subset(train, train_indices.indices)
    validation_subset = torch.utils.data.Subset(validation, val_indices.indices)
    
    sweep_train = make_sweep_train(CNNCIFAR10, train_subset, validation_subset)

    sweep_train()

if __name__ == "__main__":
    main()
