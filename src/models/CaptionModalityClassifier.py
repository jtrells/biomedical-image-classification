import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix

# TODO: There is a bug in the backbone sizes when we send a batch with only one value
class CNNTextBackbone(nn.Module):
    def __init__(self,
                 max_input_length=200,
                 vocab_size=20000,
                 embedding_dim=300,
                 filters=100,
                 embeddings=None,
                 num_classes=4,                 
                 train_embeddings=True):        
        super().__init__()
        self.num_classes = num_classes
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.embeddings.weight.requires_grad = train_embeddings

        self.conv1d_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=max_input_length - 3 + 1)

        self.conv1d_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=4, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=max_input_length - 4 + 1)

        self.conv1d_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=max_input_length - 5 + 1)

        self.dropout = nn.Dropout(0.5)        

    def forward(self, x):
        # x: (batch, max_input_length)
        x = self.embeddings(x)
        # x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1, 2)
        # x: (batch, embedding_dim, sentence_len) to match conv1D

        x1 = self.conv1d_1(x)
        # x1: (batch, filters, sentence_len - kernel_size + 1)
        x1 = self.relu1(x1)
        x1 = self.maxpool1(x1)
        # x1: (batch, filters, 1)
        x1 = x1.squeeze()

        x2 = self.conv1d_1(x)
        x2 = self.relu2(x2)
        x2 = self.maxpool2(x2)
        x2 = x2.squeeze()

        x3 = self.conv1d_3(x)
        x3 = self.relu3(x3)
        x3 = self.maxpool3(x3)
        x3 = x3.squeeze()

        x = torch.cat((x1, x2, x3), dim=1)
        # x: (batch, filters * 3)
        x = self.dropout(x)
        
        return x


class CaptionModalityClassifier(pl.LightningModule):
    def __init__(self,
                 max_input_length=200,
                 vocab_size=20000,
                 embedding_dim=300,
                 filters=100,
                 embeddings=None,
                 num_classes=4,
                 train_embeddings=True,
                 target_classes=None,
                 lr=1e-3):        
        super().__init__()
        # get these hyperparameters for free on my logger
        self.save_hyperparameters('max_input_length', 'filters', 'vocab_size', 'lr')
        self.target_classes = target_classes
        
        if embeddings is None:
            embeddings = np.random.rand(vocab_size, embedding_dim)        
        self.num_classes = num_classes
        self.CNNText = CNNTextBackbone(max_input_length=max_input_length,
                                      vocab_size=vocab_size,
                                      embedding_dim=embedding_dim,
                                      filters=filters,
                                      embeddings=embeddings,
                                      num_classes=num_classes,
                                      train_embeddings=train_embeddings)
        self.fc = nn.Linear(filters * 3, num_classes)                
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.CNNText(x)
        x = self.fc(x)        
        return x # x: (batch, num_classes)
    
    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
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
        x, _, y = batch
        y_hat = self(x)
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
        x, _, y = batch
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
                                
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_confusion_matrix(y_true.cpu(), y_pred.cpu(), ax=ax)
        self.logger.experiment.log({'confusion_matrix_test': fig}) 
        
        results = pl.EvalResult()
        results.log('test_acc', accuracy)
        return results
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    