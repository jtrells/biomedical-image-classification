import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy
import numpy as np

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
        #TODO: make default a random matrix
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
                 lr=1e-3):        
        super().__init__()
        self.save_hyperparameters('max_input_length', 'filters', 'vocab_size', 'lr')
        if embeddings is None:
            embeddings = np.random.rand(vocab_size, embedding_dim)
        
        self.num_classes = num_classes
        self.accuracy = Accuracy(num_classes)
        
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
        # x: (batch, num_classes)
        return x
    
    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)            
        acc = self.accuracy(y_hat, y)
         
        result = pl.TrainResult(loss)        
        result.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, on_epoch=True, prog_bar=True, on_step=False)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)        
        acc = self.accuracy(y_hat, y)
        
        # TODO: There is an issue open for the behavior of EvalResult and checkpoint save
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3291
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, on_epoch=True, prog_bar=True, on_step=False)        
        return result
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    