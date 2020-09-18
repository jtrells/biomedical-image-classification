import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics.classification import Accuracy
import numpy as np

class MultiModalityClassifier(pl.LightningModule):
    def __init__(self, text_model, image_model, num_classes=4):
        super().__init__()
        self.accuracy = Accuracy(num_classes)
        
        self.text_model  = text_model
        self.image_model = image_model
        self.fc = nn.Linear(300+2048, num_classes)
        # do not update model parameters
        self.text_model.freeze()
        for p in image_model.parameters():
            p.requires_grad = False
        # self.image_model.freeze() # TODO: freeze like a normal network
        # remove last layer
        self.text_model.fc = nn.Identity()
        self.image_model.fc = nn.Identity()        
   
    def forward(self, text, image):
        text = text.view(text.size(0), -1)
        text_features  = self.text_model(text)
        image_features = self.image_model(image)
        x = torch.cat([text_features, image_features], dim=1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x_text, x_image, y = batch
        y_hat = self(x_text, x_image)
        loss = F.cross_entropy(y_hat, y)            
        acc = self.accuracy(y_hat, y)
         
        result = pl.TrainResult(loss)        
        result.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, on_epoch=True, prog_bar=True, on_step=False)
        return result
    
    def validation_step(self, batch, batch_idx):
        x_text, x_image, y = batch
        y_hat = self(x_text, x_image)
        loss = F.cross_entropy(y_hat, y)        
        acc = self.accuracy(y_hat, y)
        
        result = pl.EvalResult(loss)
        result.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, on_epoch=True, prog_bar=True, on_step=False)        
        return result
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        
    