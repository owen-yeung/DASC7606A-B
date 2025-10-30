import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
import os
import logging


class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    """
    ResNet-style CNN architecture optimized for CIFAR-100
    Uses residual connections, batch normalization, and progressive channel expansion
    """

    def __init__(self, num_classes=100, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks with increasing channels: 64 -> 128 -> 256 -> 512
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, dropout_rate=dropout_rate)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def create_model(num_classes, device):
    """Create and initialize the model"""
    model = SimpleCNN(num_classes=num_classes, dropout_rate=0.2)
    model = model.to(device)
    # diagnostics: write summary and parameter counts
    try:
        log_dir = Path(os.environ.get("TRAIN_LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with (log_dir / "model_summary.txt").open('w') as f:
            f.write(str(model))
            f.write("\n\n")
            f.write(f"total_params: {total_params}\n")
            f.write(f"trainable_params: {trainable_params}\n")
            f.write(f"device: {device}\n")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to write model summary: {e}")
    return model
