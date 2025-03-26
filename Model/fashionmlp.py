import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import lightning.pytorch as pl


class FashionMNISTModelMLP(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=10, learning_rate=1e-3, seed=None):
        super().__init__()

        # Model structure: Original CNN layers
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(12 * 12 * 64, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics using torchmetrics
        self.accuracy = MulticlassAccuracy(num_classes=out_channels)
        self.precision = MulticlassPrecision(num_classes=out_channels, average='macro')
        self.recall = MulticlassRecall(num_classes=out_channels, average='macro')
        self.f1 = MulticlassF1Score(num_classes=out_channels, average='macro')

        # Set seed if specified
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Forward pass using the CNN architecture (conv1, conv2, fully connected layers).
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 12 * 12 * 64)  # Flatten the tensor before passing to fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Return raw logits (no log_softmax)
        return x

    def configure_optimizers(self):
        """
        Optimizer configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _common_step(self, batch):
        """
        Common step for training, validation, and testing.
        """
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.criterion(y_pred, labels)
        return loss, y_pred, labels

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        loss, y_pred, labels = self._common_step(batch)

        # Compute metrics
        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        f1 = self.f1(y_pred, labels)

        # Log metrics
        self.log_dict({
            "train_loss": loss,
            "train_accuracy": accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1_score": f1
        }, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        loss, y_pred, labels = self._common_step(batch)

        # Compute metrics
        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        f1 = self.f1(y_pred, labels)

        # Log metrics
        self.log_dict({
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1_score": f1
        }, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        loss, y_pred, labels = self._common_step(batch)

        # Compute metrics
        accuracy = self.accuracy(y_pred, labels)
        precision = self.precision(y_pred, labels)
        recall = self.recall(y_pred, labels)
        f1 = self.f1(y_pred, labels)

        # Log metrics
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1
        }, on_step=False, on_epoch=True, prog_bar=False)

        return loss
