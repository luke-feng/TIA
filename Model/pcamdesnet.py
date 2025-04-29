import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchvision.models as models
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class PCAMDenseNet121(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=2, learning_rate=1e-3, seed=None):
        super().__init__()
        self.save_hyperparameters()

        # 加载预训练 DenseNet121
        self.model = models.densenet121(pretrained=True)

        # 修改第一层卷积（DenseNet默认接受 3 通道，可以省略）
        if in_channels != 3:
            self.model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换分类头
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, out_channels)

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = MulticlassAccuracy(num_classes=out_channels)
        self.precision = MulticlassPrecision(num_classes=out_channels, average='macro')
        self.recall = MulticlassRecall(num_classes=out_channels, average='macro')
        self.f1 = MulticlassF1Score(num_classes=out_channels, average='macro')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

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
