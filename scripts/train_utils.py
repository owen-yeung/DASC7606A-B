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

def load_transforms(split: str = "test", policy: str = "randaugment"):
    """
    Load the data transformations for a given split.
    Default to 'test' to remain compatible with main.py's evaluation call.
    """
    if split == "train":
        aug = []
        aug.append(transforms.RandomCrop(32, padding=4))
        aug.append(transforms.RandomHorizontalFlip())
        if policy == "autoaugment":
            try:
                aug.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            except AttributeError:
                pass
        else:
            try:
                aug.append(transforms.RandAugment(num_ops=2, magnitude=11))
            except AttributeError:
                pass
        aug.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.4),
        ])
        return transforms.Compose(aug)
    # test/val
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

def load_data(root_dir: str, batch_size: int, num_workers: int = None, policy: str = "randaugment") -> Tuple[DataLoader, DataLoader]:
    """
    Load data for training/validation.
    - If 'root_dir' points to an ImageFolder root, split 80/20 and use on-the-fly transforms.
    - Otherwise, fallback to TorchVision CIFAR-100 datasets under 'root_dir'.

    Args:
        root_dir: Either ImageFolder root (augmented) or TorchVision base directory
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

    pin = torch.cuda.is_available()

    # Determine if root_dir is an ImageFolder directory (classes as subfolders)
    if os.path.isdir(root_dir):
        # Use ImageFolder with separate instances for train/val transforms
        ds_train_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("train", policy))
        ds_val_full = datasets.ImageFolder(root=root_dir, transform=load_transforms("test", policy))
        # Consistent split indices
        total_len = len(ds_train_full)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        generator = torch.Generator()
        train_subset, val_subset = random_split(range(total_len), [train_size, val_size], generator=generator)
        # Apply indices to respective datasets
        from torch.utils.data import Subset
        train_dataset = Subset(ds_train_full, train_subset.indices)
        val_dataset = Subset(ds_val_full, val_subset.indices)
    else:
        # Fallback to TorchVision CIFAR-100 datasets
        train_dataset = datasets.CIFAR100(root=root_dir, train=True, download=True, transform=load_transforms("train", policy))
        val_dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=load_transforms("test", policy))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )

    print(f"CIFAR-100 loaded under: {root_dir}")
    try:
        tr_len = len(train_dataset)
    except Exception:
        tr_len = "unknown"
    try:
        va_len = len(val_dataset)
    except Exception:
        va_len = "unknown"
    print(f"Training set size: {tr_len} | Validation set size: {va_len}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float, t_max: int = 100):
    """
    Define the loss function and optimizer
    Args:
        model: The model to train
        lr: Learning rate
        weight_decay: Weight decay
        t_max: Unused; kept for backward compatibility if needed
    Returns:
        criterion: The loss function
        optimizer: The optimizer
        scheduler: The scheduler
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    # Keep ReduceLROnPlateau for compatibility with main.py's scheduler.step(val_loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
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

    use_cuda = torch.cuda.is_available() and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    for inputs, labels in progress_bar:
        if use_cuda:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

        for inputs, labels in progress_bar:
            if torch.cuda.is_available() and device == "cuda":
                with torch.cuda.amp.autocast():
                    inputs = inputs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
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
