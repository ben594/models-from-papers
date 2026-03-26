from pathlib import Path
from typing import Any

import torch
from loaders import DATASET_NAME, create_loaders
from model import ViT
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_LR = 8e-4
NUM_EPOCHS = 5


def prepare_data_dir(data_dir: str = "data") -> Path:
    dataset_root = Path(data_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    return dataset_root


def get_train_loader(data_dir: Path) -> DataLoader[Any]:
    train_loader, _ = create_loaders(data_dir, "train")
    return train_loader


def get_test_loader(data_dir: Path) -> DataLoader[Any]:
    test_loader, _ = create_loaders(data_dir, "test")
    return test_loader


def get_classes(data_dir: Path) -> list[str]:
    _, classes = create_loaders(data_dir, "train")
    return classes


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch {epoch}/{num_epochs} [train]",
        leave=False,
    )

    for batch_idx, (inputs, targets) in enumerate(progress_bar, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
        progress_bar.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
            batch=batch_idx,
        )

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch {epoch}/{num_epochs} [eval]",
        leave=False,
    )

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(progress_bar, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += batch_size
            progress_bar.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                acc=f"{total_correct / total_examples:.4f}",
                batch=batch_idx,
            )

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def train(
    model: nn.Module,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
) -> None:
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
        )
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )


def main() -> None:
    data_dir = prepare_data_dir(str(DATA_DIR))
    train_loader = get_train_loader(data_dir)
    test_loader = get_test_loader(data_dir)
    classes = get_classes(data_dir)

    device = get_device()
    model = ViT(n_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=BASE_LR)

    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Classes: {classes}")

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )


if __name__ == "__main__":
    main()
