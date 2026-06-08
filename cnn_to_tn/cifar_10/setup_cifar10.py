from pathlib import Path
from torchvision import datasets, transforms
import logging

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(REPO_ROOT / "data" / "cifar10")

def download_cifar10():
    """Download CIFAR-10 train and test sets."""
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    logging.info(f"CIFAR-10 successfully cached at {DATA_PATH}")

if __name__ == "__main__":
    download_cifar10()
