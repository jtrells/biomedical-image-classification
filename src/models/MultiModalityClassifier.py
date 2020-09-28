import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix

class MultiModalityClassifier(pl.LightningModule):
    def __init__(self, text_model, image_model, num_filters=100, num_vision_outputs=1024, num_classes=4, cm="test", lr=1e-4):
        super().__init__()
        self.cm = cm
        
        self.save_hyperparameters('num_filters', 'lr', 'num_vision_outputs')
        
        self.text_model  = text_model
        self.image_model = image_model
#         self.fc = nn.Linear(num_filters*3 + num_vision_outputs, num_classes)
        self.fc = nn.Linear(num_filters*3 + num_vision_outputs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        # do not update model parameters
        #self.text_model.freeze()
        #for p in image_model.parameters():
        #    p.requires_grad = False
        # self.image_model.freeze() # TODO: freeze like a normal network
        # remove last layer
#         if not is_multilabel:
        #self.text_model.fc = nn.Identity() # comment for new
        self.image_model.fc = nn.Identity()     
        self.dropout = nn.Dropout(p=0.4)
   
    def forward(self, text, image):
#         text = text.view(text.size(0), -1)
#         text_features  = self.text_model(text)
        text_features = self.text_model.features(text)
#         x1 = self.dropout(text_features) 
        image_features = self.image_model(image)
#         x2 = self.dropout(image_features) 
        
#         x = torch.cat([x1, x2], dim=1)        
        x = torch.cat([text_features, image_features], dim=1)        
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x_text, x_image, y = batch
        y_hat = self(x_text, x_image)
        loss = F.cross_entropy(y_hat, y)                    
         
        _, preds = torch.max(y_hat, dim=1)
        acc = 100 * torch.sum(preds == y.data) / (y.shape[0] * 1.0)       
        return {'loss': loss, 'train_acc': acc}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'].float() for x in outputs]).mean()        
        logs = {'train_loss': avg_loss, 'train_acc': avg_acc}
        return {'avg_train_loss': avg_loss, 'avg_train_acc': avg_acc, 'log': logs, 'progress_bar': logs}    
    
    def validation_step(self, batch, batch_idx):
        x_text, x_image, y = batch
        y_hat = self(x_text, x_image)
        loss = F.cross_entropy(y_hat, y)
        
        _, preds = torch.max(y_hat, dim=1)
        acc = 100 * torch.sum(preds == y.data) / (y.shape[0] * 1.0)        
        return {'val_loss': loss, 'val_acc': acc}    
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in outputs]).mean()
        
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': logs, 'progress_bar': logs}    
    
    def test_step(self, batch, batch_idx):
        x_text, x_image, y = batch
        y_hat = self(x_text, x_image)
        
        result = pl.EvalResult()
        result.y_hat = y_hat
        result.y = y
        return result

    def test_epoch_end(self, outputs):        
        y_true = outputs.y
        _, y_pred = torch.max(outputs.y_hat, dim=1)        
        accuracy = 100 * torch.sum(y_pred == y_true.data) / (y_true.shape[0] * 1.0)
        print("Accuracy: " + str(accuracy.item()))
                                
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_confusion_matrix(y_true.cpu(), y_pred.cpu(), ax=ax)
        if self.logger:
            self.logger.experiment.log({'confusion_matrix_{0}'.format(self.cm): fig}) 
        
        results = pl.EvalResult()
        results.log('test_acc', accuracy)
        return results    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
    