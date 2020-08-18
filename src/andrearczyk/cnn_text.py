import torch.nn as nn
import torch
import numpy as np


class Conv1dBlock(nn.Module):
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
        # x_out shape: (batch_size, num_filters)
        x = self.block(x)
        return x.squeeze()


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

    def __init__(self, max_input_length, max_words, dimension, embeddings, num_classes=31, train_embeddings=True, num_filters=100):
        super(Cnn1dText, self).__init__()

        self.embedding = nn.Embedding(max_words, dimension)
        if embeddings is None:
            embeddings = np.random.rand(max_words, dimension)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = train_embeddings

        self.conv1_block1 = Conv1dBlock(dimension, num_filters, 1, max_input_length)
        self.conv1_block2 = Conv1dBlock(dimension, num_filters, 2, max_input_length)
        self.conv1_block3 = Conv1dBlock(dimension, num_filters, 3, max_input_length)
        self.conv1_block4 = Conv1dBlock(dimension, num_filters, 4, max_input_length)
        self.conv1_block5 = Conv1dBlock(dimension, num_filters, 5, max_input_length)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 5, num_classes)


    def forward(self, x):
        # x: (batch, max_input_length)
        x = self.embedding(x)
        # x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1, 2)
        # x: (batch, embedding_dim, sentence_len) to match conv1D

        x1 = self.conv1_block1(x)
        x2 = self.conv1_block2(x)
        x3 = self.conv1_block3(x)
        x4 = self.conv1_block4(x)
        x5 = self.conv1_block5(x)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # x: (batch, filters * 5)
        x = self.dropout(x)
        # x_out: (batch, 1)
        return self.fc(x)
