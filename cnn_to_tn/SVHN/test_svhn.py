from torchvision import datasets, transforms
from torch import nn
import torch
import torch.optim as optim

from SVHN.svhn_model import CNNSVHN
from utils.train import train_model
from SVHN.setup_svhn import DATA_PATH


def main():
    config = {
        "num_stages": 4,
        "blocks_per_stage": 3,
        "upscale_factor": 2,
        "channels_factor": 2,
        "start_output_channels": 32,
        "kernel_size": 3,
        "learning_rate": 1e-4,
        "momentum": 0.9,
        "rms_norm_mode": "global",
        "optimizer": "adamw",
        "weight_decay": 5e-4,
        "batch_size": 128,
        "label_smoothing": 0.05,
        "repeat_factor": 2,
        "epochs": 50,
        "warmup_epochs": 10,
        "cutmix_alpa": 0.5,
        "mixup_alpha": 0.1,
    }

    # SVHN normalization stats
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

    # Train on FULL train + extra splits (no validation split)
    train_main_dataset = datasets.SVHN(
        root=data_path,
        split="train",
        download=True,
        transform=train_transformation,
    )

    extra_dataset = datasets.SVHN(
        root=data_path,
        split="extra",
        download=True,
        transform=train_transformation,
    )

    train_dataset = torch.utils.data.ConcatDataset([train_main_dataset, extra_dataset])

    # Held-out test set for final evaluation
    test_dataset = datasets.SVHN(
        root=data_path,
        split="test",
        download=True,
        transform=evaluation_transformation,
    )

    print("Training samples (train + extra):", len(train_dataset))
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

    model = CNNSVHN(config)

    train_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        optimizer_arg=optimizer_arg,
        criterion_arg=nn.CrossEntropyLoss,
        tuning_flag=False,
        project_name="Final-models",
        run_name="SVHN-final",
    )


if __name__ == "__main__":
    main()
