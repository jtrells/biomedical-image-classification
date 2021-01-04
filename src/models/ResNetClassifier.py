import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchvision import models
from torch import nn

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_confusion_matrix

class ResNet(pl.LightningModule):
    def __init__(self,
                 name,
                 num_classes,
                 pretrained=True,
                 fine_tuned_from="fc",
                 lr=1e-3,
                 class_weights=None):
        super().__init__()

        self.save_hyperparameters("name", "num_classes", "pretrained", "fine_tuned_from", "lr", "class_weights")   
        self.model = self._get_resnet_model() # load resnet_x model
        self.set_fine_tuning()                # set requires_grad values
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.hparams.class_weights))

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
    
    def _get_resnet_block(self, blk_num):
        if blk_num == 4:
            return self.model.layer4
        elif blk_num == 3:
            return self.model.layer3
        elif blk_num == 2:
            return self.model.layer2
        elif blk_num == 1:
            return self.model.layer1
        raise Exception(f"Resnet does not have layer {blk_num}")

    def _unfreeze_block(self, blk_num, layer_num):
        block = self._get_resnet_block(blk_num)
        if len(block) <= layer_num:
            raise Exception(f"Block {blk_num} does not have {layer_num+1} layers")
        # fine-tune everything from the given block-layer all the way up in block
        for idx in range(len(block) - layer_num):
            for layer in block[-(idx+1)].parameters():
                layer.requires_grad = True

    def set_fine_tuning(self):
        # by default set all to false
        for p in self.model.parameters(): p.requires_grad = False
        # case for retraining everything
        if self.hparams.fine_tuned_from == "whole":
            for p in self.model.parameters(): p.requires_grad = True
            return

        # always train parameters in fc
        for p in self.model.fc.parameters(): p.requires_grad = True
        if self.hparams.fine_tuned_from == "fc": return

        # start fine tuning previous layers and blocks
        blk_num, layer_num = self.hparams.fine_tuned_from.split('-')
        blk_num = int(blk_num)
        layer_num = int(layer_num)

        for block_idx in [4, 3, 2, 1]:
            # unfreeze from earlier? then unfreeze all the block   
            layer_k_num = layer_num if blk_num == block_idx else 0
            self._unfreeze_block(block_idx, layer_k_num)
            # leave previous blocks untouched
            if block_idx == blk_num: return

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {'loss': loss, 'train_preds': preds, 'train_trues': y}

    def training_epoch_end(self, outputs):
        y_preds = torch.stack([vec.float() for x in outputs for vec in x['train_preds']])
        y_trues = torch.stack([vec.float() for x in outputs for vec in x['train_trues']])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        acc = 100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        logs = { 'train_loss': avg_loss, 'train_acc': acc}
        return { 'train_avg_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        _, preds = torch.max(y_hat, dim=1)
        return {'val_loss': loss, 'val_preds': preds, 'val_trues': y}        
        
    def validation_epoch_end(self, outputs):        
        y_preds = torch.stack([vec.float() for x in outputs for vec in x['val_preds']])
        y_trues = torch.stack([vec.float() for x in outputs for vec in x['val_trues']])
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()        

        acc = 100 * torch.sum(y_preds == y_trues.data) / (y_trues.shape[0] * 1.0)
        logs = { 'val_loss': avg_loss, 'val_acc': acc}
        return { 'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)
        
        result = pl.EvalResult()
        result.y_hat = y_hat
        result.y = y
        return result

    def test_epoch_end(self, outputs):        
        y_true = outputs.y
        _, y_pred = torch.max(outputs.y_hat, dim=1)        
        accuracy = 100 * torch.sum(y_pred == y_true.data) / (y_true.shape[0] * 1.0)
        print("Accuracy: " + str(accuracy.item()))
        
        print(classification_report(y_true.cpu(), y_pred.cpu()))
                                
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_confusion_matrix(y_true.cpu(), y_pred.cpu(), ax=ax)
        if self.logger:
            self.logger.experiment.log({'confusion_matrix_{0}'.format(self.cm): fig}) 
        
        results = pl.EvalResult()
        results.log('test_acc', accuracy)
        return results        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)