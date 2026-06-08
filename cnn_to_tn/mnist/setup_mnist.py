from pathlib import Path
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(REPO_ROOT / "data" / "MNIST")

def download_mnist():
    """Download MNIST train and test sets."""
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

    datasets.MNIST(
        root=DATA_PATH,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    datasets.MNIST(
        root=DATA_PATH,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

if __name__ == "__main__":
    download_mnist()
