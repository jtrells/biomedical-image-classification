""" Resnet wrapper """

import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)


class ResNetClass(pl.LightningModule):
    """Lightning ResNet wrapper with support to ResNet18, 34, 50, 101, 152"""

    def __init__(
        self,
        name="resnet18",
        num_classes=6,
        pretrained=True,
        fine_tuned_from="whole",
        lr=1e-3,
        metric_monitor="val_avg_loss",
        mode_scheduler="min",
        class_weights=None,
        mean_dataset=None,
        std_dataset=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            "name",
            "num_classes",
            "pretrained",
            "fine_tuned_from",
            "lr",
            "class_weights",
            "metric_monitor",
            "mode_scheduler",
            "mean_dataset",
            "std_dataset",
        )
        self.model = self._get_resnet_model()
        self.set_fine_tuning()

    def _get_resnet_model(self):
        if self.hparams.name == "resnet18":
            model = models.resnet18(pretrained=self.hparams.pretrained)
        elif self.hparams.name == "resnet34":
            model = models.resnet34(pretrained=self.hparams.pretrained)
        elif self.hparams.name == "resnet50":
            model = models.resnet50(pretrained=self.hparams.pretrained)
        elif self.hparams.name == "resnet101":
            model = models.resnet101(pretrained=self.hparams.pretrained)
        elif self.hparams.name == "resnet152":
            model = models.resnet152(pretrained=self.hparams.pretrained)
        # retarget the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.hparams.num_classes)
        return model

    def set_fine_tuning(self):
        """Freeze the layers to update"""
        # by default set all to false
        for param in self.model.parameters():
            param.requires_grad = False
        # case for retraining everything
        if self.hparams.fine_tuned_from == "whole":
            for param in self.model.parameters():
                param.requires_grad = True
            return

        # always train parameters in the fully connected layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
        # TODO: this seems like a mistake
        if self.hparams.fine_tuned_from == "fc":
            return

    def _get_cost_function(self):
        if self.hparams.class_weights is not None:
            return nn.CrossEntropyLoss(
                weight=torch.Tensor(self.hparams.class_weights).to("cuda")
            )
        else:
            return nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass in the model"""
        out = self.model(x)
        return out

    def feature_extraction(self):
        features = nn.Sequential(*list(self.model.children())[:-1])
        return features

    def configure_optimizers(self):
        if self.hparams.mode_scheduler is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            return optimizer
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.mode_scheduler,
                patience=3,  # Patience for the Scheduler
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.hparams.metric_monitor,
            }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = self._get_cost_function()
        loss = criterion(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {"loss": loss, "train_preds": preds, "train_trues": y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_preds = torch.stack(
            [vec.float() for x in outputs for vec in x["train_preds"]]
        )
        y_trues = torch.stack(
            [vec.float() for x in outputs for vec in x["train_trues"]]
        )

        acc_train_epoch = (
            100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        )
        self.log(
            "train_acc", acc_train_epoch, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_avg_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = self._get_cost_function()
        loss = criterion(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {"loss": loss, "val_preds": preds, "val_trues": y}

    def validation_epoch_end(self, outputs):
        # This count the sanity check pass, that's why start getting some values for validation
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_preds = torch.stack([vec.float() for x in outputs for vec in x["val_preds"]])
        y_trues = torch.stack([vec.float() for x in outputs for vec in x["val_trues"]])

        acc_val_epoch = (
            100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        )
        self.log("val_acc", acc_val_epoch, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_avg_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        print(f"acc_val_epoch: {acc_val_epoch},val_avg_loss: {avg_loss} ")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = self._get_cost_function()
        loss = criterion(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {"loss": loss, "test_preds": preds, "test_trues": y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_preds = torch.stack([vec.float() for x in outputs for vec in x["test_preds"]])
        y_trues = torch.stack([vec.float() for x in outputs for vec in x["test_trues"]])
        accuracy = 100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        print("Accuracy: " + str(accuracy.item()))

        print(classification_report(y_trues.cpu(), y_preds.cpu()))

        fig, ax = plt.subplots(figsize=(4, 4))
        plot_confusion_matrix(y_trues.cpu(), y_preds.cpu(), ax=ax)
        if self.logger:
            self.logger.experiment.log({"confusion_matrix": fig})

        self.log("test_acc", accuracy)
        self.log("test_loss", avg_loss)
        self.log(
            "Macro F1-Score", f1_score(y_trues.cpu(), y_preds.cpu(), average="macro")
        )
        self.log(
            "Balanced Accuracy", balanced_accuracy_score(y_trues.cpu(), y_preds.cpu())
        )
        self.log(
            "Macro Recall", recall_score(y_trues.cpu(), y_preds.cpu(), average="macro")
        )
        self.log(
            "Macro Precision",
            precision_score(y_trues.cpu(), y_preds.cpu(), average="macro"),
        )
