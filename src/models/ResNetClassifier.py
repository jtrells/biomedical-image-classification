import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchvision import models
from torch import nn


class ResNet(pl.LightningDataModule):
    def __init__(self, name, num_classes, pretrained=True, fine_tune_from="fc", lr=1e-3):
        super().__init__()

        self.save_hyperparameters("name", "num_classes", "pretrained", "fine_tuned_from", "lr")

        self.name = name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.fine_tune_from = fine_tune_from

        self.model = self._get_resnet_model()
        self._set_fine_tuning()

    def _get_resnet_model(self):
        if self.name == "resnet18":
            model = models.resnet18(pretrained=self.pretrained)
        elif self.name == "resnet34":
            model = models.resnet34(pretrained=self.pretrained)
        elif self.name == "resnet50":
            model = models.resnet50(pretrained=self.pretrained)
        elif self.name == "resnet101":
            model = models.resnet101(pretrained=self.pretrained)
        elif self.name == "resnet152":
            model = models.resnet152(pretrained=self.pretrained)   
        # retarget the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)        

        return model
    
    def _get_resnet_block(self, layer_num):
        if layer_num == 4:
            return self.model.layer4
        elif layer_num == 3:
            return self.model.layer3
        elif layer_num == 2:
            return self.model.layer2
        elif layer_num == 1:
            return self.model.layer1
        raise Exception(f"Resnet does not have layer {layer_num}")

    def _unfreeze_block(self, blk_num, layer_num):
        block = self._get_resnet_block(blk_num)
        if len(block) <= layer_num:
            raise Exception(f"Block does not have {layer_num-1} layers")
        # fine-tune everything from the given block-layer all the way up in block
        for idx in range(len(block) - layer_num):
            for layer in block[-(idx+1)].parameters():
                layer.requires_grad = True

    def _set_fine_tuning(self):
        # case for retraining everything
        if self.fine_tune_from == "whole":
            for p in self.model.parameters(): p.requires_grad = True
            return

        # always train parameters in fc
        for p in self.model.fc.parameters(): p.requires_grad = True
        if self.fine_tune_from == "fc": return

        # start fine tuning previous layers and blocks
        blk_num, layer_num = self.fine_tune_from.splot('-')

        for block_idx in [4, 3, 2, 1]:
            # unfreeze from earlier? then unfreeze all the block   
            layer_k_num = layer_num if blk_num == block_idx else 0
            self._unfreeze_block(blk_num, layer_k_num)
            # leave previous blocks untouched
            if block_idx == blk_num: return

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {'loss': loss, 'train_preds': preds, 'train_trues': y}

    def training_epoch_end(self, outputs):
        y_preds = torch.stack([x['train_preds'] for x in outputs])
        y_trues = torch.stack([x['train_trues'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        acc = 100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        logs = { 'loss': avg_loss, 'train_acc': acc}
        return { 'loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {'val_loss': loss, 'val_preds': preds, 'val_trues': y}        
        
    def validation_epoch_end(self, outputs):
        y_preds = torch.stack([x['val_preds'] for x in outputs])
        y_trues = torch.stack([x['val_trues'] for x in outputs])
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        acc = 100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        logs = { 'val_loss': avg_loss, 'val_acc': acc}
        return { 'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)