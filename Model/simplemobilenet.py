import torch
import lightning as pl
from torch import nn
import torchmetrics

class SimpleMobileNet(pl.LightningModule):
    def __init__(
            self,
            in_channels=3,
            out_channels=10,
            learning_rate=1e-3,
            seed=None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Define MobileNet layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_channels)

        # Define metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=out_channels, average='macro')
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=out_channels, average='macro')
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=out_channels, average='macro')

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

    def _common_step(self, batch):
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.loss_fn(y_pred, labels)
        return loss, y_pred, labels

    def training_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)

        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "train_loss": loss,
            "train_accuracy": accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels}

    def validation_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)

        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels}

    def test_step(self, batch, batch_idx):
        loss, y_pred, labels = self._common_step(batch)
        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        F1 = self.F1(y_pred, labels)
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": F1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": y_pred, "y": labels}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

