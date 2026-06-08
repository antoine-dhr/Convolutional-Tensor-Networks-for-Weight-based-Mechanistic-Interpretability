from torchvision import datasets, transforms
import torch
import wandb
import torch.optim as optim

from mnist.mnist_model import CNNMNIST
from mnist.setup_mnist import DATA_PATH
from utils.tuning import make_sweep_train, get_best_run
from utils.train import train_model

SPLIT_SEED = 42


def main():
    # MNIST has a single channel; mean/std computed over the training set
    train_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    evaluation_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_path = DATA_PATH

    train = datasets.MNIST(
        root=data_path,
        train=True,
        download=False,
        transform=train_transformation,
    )

    validation = datasets.MNIST(
        root=data_path,
        train=True,
        download=False,
        transform=evaluation_transformation,
    )

    test = datasets.MNIST(
        root=data_path,
        train=False,
        download=False,
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

    # MNIST is simple enough that MixUp/CutMix is not beneficial
    sweep_train = make_sweep_train(CNNMNIST, train_subset, validation_subset, apply_mixup=False)

    sweep_train()


if __name__ == "__main__":
    main()
