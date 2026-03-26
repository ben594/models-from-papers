from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_DIM = 224
BATCH_SIZE = 32
DATASET_NAME = "cifar10"


def create_loaders(data_dir: Path, split: str) -> tuple[DataLoader, list[str]]:
    if split not in {"train", "test"}:
        raise ValueError("Dataloader split must be train or test")

    is_train = split == "train"
    transform = transforms.Compose(
        [transforms.Resize((IMAGE_DIM, IMAGE_DIM)), transforms.ToTensor()]
    )

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=is_train,
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train)
    classes = list(dataset.classes)
    return dataloader, classes
