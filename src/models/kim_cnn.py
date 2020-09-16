import torch.nn as nn
import torch
import numpy as np


class Conv1dBlock(nn.Module):
    # L2 regularization used by Kim can be applied when defining the loss function.
    # Keras allows the setting to be part of the layer, for Pytorch check
    # https://stackoverflow.com/questions/44641976/in-pytorch-how-to-add-l1-regularizer-to-activations
    def __init__(self, embedding_dimension, num_filters, kernel_size, max_input_length):
        super(Conv1dBlock, self).__init__()
        block = [nn.Conv1d(in_channels=embedding_dimension,
                           out_channels=num_filters,
                           kernel_size=kernel_size,
                           stride=1)]
        # x_out shape: (batch_size, num_filters, max_input_length - kernel_size + 1)
        block += [nn.ReLU()]
        # max pooling over the whole sentence after convolution
        block += [nn.MaxPool1d(kernel_size=max_input_length - kernel_size + 1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        # x_in  shape: (batch_size, embedding_dimension, max_input_length)
        # x_out shape: (batch_size, num_filters, 1)
        return self.block(x)


class Cnn1dText(nn.Module):
    '''
    Args:
        max_input_length (int): Maximum number of words in an input sentence
        max_words (int): Number of words in the embedding vocabulary matrix
        dimension (int): Number of embedding features per word
        embeddings (np.matrix(max_words, dimension)): Matrix of word2vec embeddings (e.g. Glove). If None, 
            generate a matrix of random numbers
        num_classes (int): Target sentiment levels, 2 for positive or negative reviews used in MR, SST uses more
            detailed levels of sentiment
        train_embeddings (boolean): Whether to fine tune the embeddings matrix or not
        filters (int): Number of filters in Conv1D blocks (Kim used 100)
    '''

    def __init__(self, max_input_length, max_words, dimension, embeddings, num_classes=2, train_embeddings=False, filters=100):
        super(Cnn1dText, self).__init__()

        self.embedding = nn.Embedding(max_words, dimension)
        if not embeddings:
            embeddings = np.random.rand(max_words, dimension)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = train_embeddings
    # TODO


class CNN1DText(nn.Module):

    def __init__(self, max_input_length, max_words, dimension, embeddings, num_classes=2, train_embeddings=False):
        super(CNN1DText, self).__init__()
        self.filters = 100

        self.embedding = nn.Embedding(max_words, dimension)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = train_embeddings

        self.conv1d_1 = nn.Conv1d(
            in_channels=dimension, out_channels=self.filters, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=max_input_length - 3 + 1)

        self.conv1d_2 = nn.Conv1d(
            in_channels=dimension, out_channels=self.filters, kernel_size=4, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=max_input_length - 4 + 1)

        self.conv1d_3 = nn.Conv1d(
            in_channels=dimension, out_channels=self.filters, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=max_input_length - 5 + 1)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.filters * 3, num_classes)

    def forward(self, x):
        # x: (batch, max_input_length)
        x = self.embedding(x)
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
        x_logits = self.fc(x)
        # x: (batch, 1)
        return x_logits
