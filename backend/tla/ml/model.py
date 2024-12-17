import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassCohenKappa,
    MulticlassAUROC,
    MulticlassConfusionMatrix
)

class SimpleTibetanNumberClassifier(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.l1 = torch.nn.Linear(28 * 28, num_classes)

        # Defining the tensor metrics from the torch library
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_precision_metrics = MulticlassPrecision(num_classes=self.num_classes)
        self.val_recall_metrics = MulticlassRecall(num_classes=self.num_classes)
        self.val_auroc_metrics = MulticlassAUROC(num_classes=self.num_classes)
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize=None)
        self.val_f1_metrics = MulticlassF1Score(num_classes=self.num_classes)
        self.val_cohen_kappa_metrics = MulticlassCohenKappa(num_classes=self.num_classes)

        # Test metrics
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_precision_metrics = MulticlassPrecision(num_classes=self.num_classes)
        self.test_recall_metrics = MulticlassRecall(num_classes=self.num_classes)
        self.test_auroc_metrics = MulticlassAUROC(num_classes=self.num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize=None)
        self.test_f1_metrics = MulticlassF1Score(num_classes=self.num_classes)
        self.test_cohen_kappa_metrics = MulticlassCohenKappa(num_classes=self.num_classes)

    def forward(self, x):
        return self.l1(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.softmax(logits, dim=1)

        # Update metrics
        self.val_accuracy.update(preds, y)
        self.val_precision_metrics.update(preds, y)
        self.val_recall_metrics.update(preds, y)
        self.val_auroc_metrics.update(preds, y)
        self.val_confusion_matrix.update(preds, y)
        self.val_f1_metrics.update(preds, y)
        self.val_cohen_kappa_metrics.update(preds, y)

        # logging
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_precision", self.val_precision_metrics, prog_bar=True)
        self.log("val_recall", self.val_recall_metrics, prog_bar=True)
        self.log("val_auroc", self.val_auroc_metrics, prog_bar=True)
        self.log("val_f1", self.val_f1_metrics, prog_bar=True)
        self.log("val_cohen_kappa", self.val_cohen_kappa_metrics, prog_bar=True)
        return loss 
      
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.softmax(logits, dim=1)

        # Update test metrics
        self.test_accuracy.update(preds, y)
        self.test_precision_metrics.update(preds, y)
        self.test_recall_metrics.update(preds, y)
        self.test_auroc_metrics.update(preds, y)
        self.test_confusion_matrix.update(preds, y)
        self.test_f1_metrics.update(preds, y)
        self.test_cohen_kappa_metrics.update(preds, y)

        # Log test metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        self.log("test_precision", self.test_precision_metrics, prog_bar=True)
        self.log("test_recall", self.test_recall_metrics, prog_bar=True)
        self.log("test_auroc", self.test_auroc_metrics, prog_bar=True)
        self.log("test_f1", self.test_f1_metrics, prog_bar=True)
        self.log("test_cohen_kappa", self.test_cohen_kappa_metrics, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
