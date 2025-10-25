import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
import csv
import logging
from pathlib import Path

# lightweight diagnostics to disk
_EPOCH_COUNTER = 0
_LOG_DIR = Path(os.environ.get("TRAIN_LOG_DIR", "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_TRAIN_CSV = _LOG_DIR / "metrics_train.csv"
_VAL_CSV = _LOG_DIR / "metrics_val.csv"
_EPOCH_CSV = _LOG_DIR / "metrics_epoch.csv"
_DATASET_SUMMARY = _LOG_DIR / "dataset_summary.txt"

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / "train.log")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    _logger.addHandler(fh)

def _append_csv(path: Path, row: dict, fieldnames: list):
    new_file = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def load_transforms(is_training=True):
    """
    Load the data transformations optimized for CIFAR-100
    
    Args:
        is_training: Whether to apply training augmentations
    """
    # CIFAR-100 mean and std
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])

def load_data(data_dir, batch_size):
    """
    Load the data from the data directory and split it into training and validation sets
    This function is similar to the cell 2. Data Preparation in 04_model_training.ipynb

    Args:
        data_dir: The directory to load the data from
        batch_size: The batch size to use for the data loaders
    Returns:
        train_loader: The training data loader
        val_loader: The validation data loader
    """
    # Define data transformations: resize, convert to tensor, and normalize
    train_transforms = load_transforms(is_training=True)
    val_transforms = load_transforms(is_training=False)

    # Load the train dataset from the augmented data directory
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Load the validation dataset from the raw data directory
    val_dataset = datasets.ImageFolder(root=data_dir + "/../../raw/val", transform=val_transforms)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)

    # Print dataset summary
    print(f"Dataset loaded from: {data_dir}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Also persist a dataset summary for diagnostics
    try:
        with _DATASET_SUMMARY.open('w') as f:
            f.write(f"data_dir: {data_dir}\n")
            f.write(f"num_classes: {len(train_dataset.classes)}\n")
            f.write(f"classes: {train_dataset.classes}\n")
            f.write(f"train_size: {len(train_dataset)}\n")
            f.write(f"val_size: {len(val_dataset)}\n")
            f.write(f"batch_size: {batch_size}\n")
    except Exception as e:
        _logger.warning(f"Failed writing dataset summary: {e}")

    return train_loader, val_loader


def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """
    Backward-compatible optimizer setup expected by main.py.
    Returns (criterion, optimizer, scheduler).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    _logger.info(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
    _logger.info("Scheduler: ReduceLROnPlateau(patience=3, factor=0.5)")
    if torch.cuda.is_available():
        _logger.info(f"CUDA available: True, device_name={torch.cuda.get_device_name(0)}")
    else:
        _logger.info("CUDA available: False")
    return criterion, optimizer, scheduler


def define_loss_and_optimizer_advanced(model: nn.Module, lr: float, weight_decay: float, 
                            num_epochs: int, steps_per_epoch: int, label_smoothing: float = 0.1):
    """
    Advanced optimizer setup with cosine warmup and mixed precision scaler.
    Returns (criterion, optimizer, scheduler, scaler).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, 
                           betas=(0.9, 0.999), eps=1e-8)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    _logger.info(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
    _logger.info(f"Label smoothing: {label_smoothing}")
    _logger.info(f"Scheduler: Cosine with warmup_steps={warmup_steps}, total_steps={total_steps}")
    if torch.cuda.is_available():
        _logger.info(f"CUDA available: True, device_name={torch.cuda.get_device_name(0)}")
    else:
        _logger.info("CUDA available: False")
    return criterion, optimizer, scheduler, scaler


def train_epoch_advanced(model, dataloader, criterion, optimizer, scheduler, scaler, device, 
               max_grad_norm=1.0, use_amp=True):
    """
    Train the model for one epoch with mixed precision and gradient clipping
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Whether to use automatic mixed precision
    Returns:
        Average loss and accuracy for the epoch
    """
    global _EPOCH_COUNTER
    _EPOCH_COUNTER += 1
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_epoch_time = time.time()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_start = time.time()

        # Zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Update scheduler
        scheduler.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%", 
             "LR": f"{current_lr:.6f}"}
        )

        # Write batch diagnostics
        batch_time = time.time() - batch_start
        _append_csv(
            _TRAIN_CSV,
            {
                "epoch": _EPOCH_COUNTER,
                "loss": float(loss.item()),
                "acc": float(100.0 * correct / max(1, total)),
                "lr": float(current_lr),
                "grad_norm": float(getattr(grad_norm, 'item', lambda: grad_norm)()),
                "batch_time_sec": float(batch_time),
                "batch_size": int(inputs.size(0)),
            },
            fieldnames=["epoch", "loss", "acc", "lr", "grad_norm", "batch_time_sec", "batch_size"],
        )

        # NaN/Inf checks
        if not torch.isfinite(loss):
            _logger.error("Non-finite loss detected. Stopping training.")
            break

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    epoch_time = time.time() - start_epoch_time

    # Persist epoch metrics
    _append_csv(
        _EPOCH_CSV,
        {
            "epoch": _EPOCH_COUNTER,
            "train_loss": float(epoch_loss),
            "train_acc": float(epoch_acc),
            "epoch_time_sec": float(epoch_time),
        },
        fieldnames=["epoch", "train_loss", "train_acc", "epoch_time_sec"],
    )
    _logger.info(f"Epoch {_EPOCH_COUNTER} train: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%, time={epoch_time:.1f}s")

    return epoch_loss, epoch_acc


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Backward-compatible training loop expected by main.py.
    Creates a no-op scheduler and GradScaler, then delegates to train_epoch_advanced.
    Returns (epoch_loss, epoch_acc).
    """
    # no-op per-step scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = GradScaler()
    return train_epoch_advanced(
        model, dataloader, criterion, optimizer, scheduler, scaler, device,
        max_grad_norm=1.0, use_amp=True
    )


def validate_epoch(model, dataloader, criterion, device, use_amp=True):
    """
    Validate the model with mixed precision
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        use_amp: Whether to use automatic mixed precision
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_start = time.time()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
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

            # Per-batch validation metrics (optional but helpful for debugging)
            _append_csv(
                _VAL_CSV,
                {
                    "epoch": _EPOCH_COUNTER,
                    "loss": float(loss.item()),
                    "acc": float(100.0 * correct / max(1, total)),
                    "batch_size": int(inputs.size(0)),
                },
                fieldnames=["epoch", "loss", "acc", "batch_size"],
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    val_time = time.time() - val_start

    # Append to epoch csv
    _append_csv(
        _EPOCH_CSV,
        {
            "epoch": _EPOCH_COUNTER,
            "val_loss": float(epoch_loss),
            "val_acc": float(epoch_acc),
            "val_time_sec": float(val_time),
        },
        fieldnames=["epoch", "val_loss", "val_acc", "val_time_sec"],
    )
    _logger.info(f"Epoch {_EPOCH_COUNTER} val: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%, time={val_time:.1f}s")

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
