import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchvision.models import swin_t, Swin_T_Weights


class ImageNet100SwinTransformer(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=100, learning_rate=1e-3, seed=None):
        super().__init__()

        self.save_hyperparameters()

        # 加载 Swin Transformer Tiny 模型（可选加载预训练权重）
        self.model = swin_t(weights=None)  # 可替换为 Swin_T_Weights.IMAGENET1K_V1

        # 替换分类头为 100 类输出
        self.model.head = nn.Linear(self.model.head.in_features, out_channels)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
