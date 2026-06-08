from torchvision import datasets, transforms
from torch import nn
import torch
import torch.optim as optim

from cifar_100.cifar_100_model import CNNCIFAR100
from utils.train import train_model
from cifar_100.setup_cifar100 import DATA_PATH


def main():
    # Best hyperparameters from sweep.yaml
    config = {
        "num_stages": 4,
        "blocks_per_stage": 3,
        "upscale_factor": 2,
        "channels_factor": 2,
        "start_output_channels": 32,
        "kernel_size": 3,
        "learning_rate": 1e-2,
        "momentum": 0.9,
        "optimizer": "sgd",
        "weight_decay": 0.0015,
        "batch_size": 128,
        "label_smoothing": 0.15,
        "repeat_factor": 2,
        "epochs": 150,
        "warmup_epochs": 5,
        "load_pretrained_model": True
    }

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

    # Train on the FULL training set (no validation split)
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transformation,
    )

    # Held-out test set for final evaluation
    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=evaluation_transformation,
    )

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # Convert dict to a namespace so config.attribute works
    config = type("Config", (), config)()

    optimizer_name = config.optimizer.lower()
    if optimizer_name == "sgd":
        optimizer_arg = optim.SGD
    elif optimizer_name == "adam":
        optimizer_arg = optim.Adam
    elif optimizer_name == "adamw":
        optimizer_arg = optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    model = CNNCIFAR100(config)

    train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        optimizer_arg=optimizer_arg,
        criterion_arg=nn.CrossEntropyLoss,
        tuning_flag=False,
        project_name="Final-models",
        run_name="CIFAR100-final",
    )


if __name__ == "__main__":
    main()
