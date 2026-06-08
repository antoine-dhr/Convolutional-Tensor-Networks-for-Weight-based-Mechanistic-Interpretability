from pathlib import Path
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(REPO_ROOT / "data" / "svhn")

def download_svhn():
    """Download SVHN train and test sets."""
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    datasets.SVHN(
        root=DATA_PATH,
        split="train",
        download=True,
        transform=transforms.ToTensor()
    )
    datasets.SVHN(
        root=DATA_PATH,
        split="test",
        download=True,
        transform=transforms.ToTensor()
    )

if __name__ == "__main__":
    download_svhn()
