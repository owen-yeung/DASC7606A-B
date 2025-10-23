import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from typing import Tuple


# CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def load_transforms(split: str = "train", policy: str = "randaugment"):
    """
    Load the data transformations for a given split.
    """
    if split == "train":
        aug = []
        # Traditional CIFAR augmentations
        aug.append(transforms.RandomCrop(32, padding=4))
        aug.append(transforms.RandomHorizontalFlip())
        # Policy
        if policy == "autoaugment":
            try:
                aug.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            except AttributeError:
                pass
        else:
            try:
                aug.append(transforms.RandAugment(num_ops=2, magnitude=9))
            except AttributeError:
                pass
        aug.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.25),
        ])
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

def load_data(root_dir: str, batch_size: int, num_workers: int = None, policy: str = "randaugment") -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-100 train and test sets with on-the-fly augmentations.

    Args:
        root_dir: Base data directory (torchvision will store under this path)
        batch_size: The batch size to use for the data loaders
        num_workers: DataLoader workers (defaults to half of CPUs)
        policy: Augment policy ("randaugment" | "autoaugment")
    Returns:
        train_loader, val_loader
    """
    if num_workers is None:
        try:
            num_workers = max(2, os.cpu_count() // 2)
        except Exception:
            num_workers = 2

    train_tf = load_transforms("train", policy)
    test_tf = load_transforms("test", policy)

    train_dataset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=train_tf)
    val_dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    print(f"CIFAR-100 loaded under: {root_dir}")
    print(f"Training set size: {len(train_dataset)} | Validation set size: {len(val_dataset)}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, t_max: int = 100):
    """
    Define the loss function and optimizer
    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay
        t_max: T_max for CosineAnnealingLR (typically num_epochs)
    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The scheduler
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr * 1e-3)
    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    autocast_device = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP autocast (works on CUDA/MPS; on CPU it’s a no-op)
        with torch.autocast(device_type=autocast_device, dtype=torch.float16 if autocast_device != "cpu" else torch.bfloat16, enabled=(autocast_device != "cpu")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        autocast_device = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with torch.autocast(device_type=autocast_device, dtype=torch.float16 if autocast_device != "cpu" else torch.bfloat16, enabled=(autocast_device != "cpu")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
