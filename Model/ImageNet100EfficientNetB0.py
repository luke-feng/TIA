import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.residual(x))


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res1 = BasicBlock(128, 128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.res2 = BasicBlock(512, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
import lightning.pytorch as pl
import torch.nn as nn
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class ImageNet100ResNet9(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=100, learning_rate=1e-3, seed=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNet9(in_channels=in_channels, num_classes=out_channels)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=out_channels)
        self.precision = MulticlassPrecision(num_classes=out_channels, average='macro')
        self.recall = MulticlassRecall(num_classes=out_channels, average='macro')
        self.f1 = MulticlassF1Score(num_classes=out_channels, average='macro')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _common_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        self.log_dict({
            "train_loss": loss,
            "train_accuracy": self.accuracy(y_hat, y),
            "train_precision": self.precision(y_hat, y),
            "train_recall": self.recall(y_hat, y),
            "train_f1_score": self.f1(y_hat, y),
        }, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        self.log_dict({
            "val_loss": loss,
            "val_accuracy": self.accuracy(y_hat, y),
            "val_precision": self.precision(y_hat, y),
            "val_recall": self.recall(y_hat, y),
            "val_f1_score": self.f1(y_hat, y),
        }, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": self.accuracy(y_hat, y),
            "test_precision": self.precision(y_hat, y),
            "test_recall": self.recall(y_hat, y),
            "test_f1_score": self.f1(y_hat, y),
        }, on_step=False, on_epoch=True)
        return loss