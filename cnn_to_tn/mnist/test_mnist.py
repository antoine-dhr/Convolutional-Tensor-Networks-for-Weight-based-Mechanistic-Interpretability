from torchvision import datasets, transforms
from torch import nn
import torch
import torch.optim as optim

from mnist.mnist_model import CNNMNIST
from mnist.setup_mnist import DATA_PATH
from utils.train import train_model


def main():
    config = {
        "num_stages": 2,
        "blocks_per_stage": 2,
        "upscale_factor": 3,
        "channels_factor": 3,
        "start_output_channels": 16,
        "kernel_size": 3,
        "learning_rate": 1e-2,
        "momentum": 0.9,
        "optimizer": "sgd",
        "weight_decay": 0, 
        "batch_size": 128,
        "label_smoothing": 0,
        "repeat_factor": 2,
        "epochs": 7,
        "warmup_epochs": 3,
        "rms_norm_mode": "global",
    }

    print(config)

    train_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    evaluation_transformation = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_path = DATA_PATH

    # Train on the FULL training set (no validation split)
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=train_transformation,
    )

    # Held-out test set for final evaluation
    test_dataset = datasets.MNIST(
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

    model = CNNMNIST(config)

    train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        optimizer_arg=optimizer_arg,
        criterion_arg=nn.CrossEntropyLoss,
        tuning_flag=False,
        project_name="Final-models",
        run_name="MNIST-final",
        apply_mixup=False,
    )


if __name__ == "__main__":
    main()
