from pathlib import Path
from torchvision import datasets, transforms
import logging

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(REPO_ROOT / "data" / "cifar100")

def download_cifar100():
    """Download CIFAR-100 train and test sets."""
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    datasets.CIFAR100(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    logging.info(f"CIFAR-100 successfully cached at {DATA_PATH}")

if __name__ == "__main__":
    download_cifar100()
