import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . \n'
    becomes
    "the rock is destined to be the 21st century 's new conan and that he 's going to make a splash even greater than arnold schwarzenegger , jean claud van damme or steven segal"
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_embedding_matrix(GLOVE_DIR, embedding_size):
    if embedding_size not in [50, 100, 200, 300]:
        print("Glove vector size should be 50, 200, 100 or 300 dimension")
        return None

    embeddings_idx = {}
    fname = 'glove.6B.%dd.txt' % embedding_size

    with open(os.path.join(GLOVE_DIR, fname)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_idx[word] = coefficients

    print('Dimension: %d; found %s word vectors.' %
          (embedding_size, len(embeddings_idx)))
    return embeddings_idx


def load_glove_matrix(GLOVE_DIR, embedding_size, vocab_size, word_index):
    glove_dict = load_embedding_matrix(GLOVE_DIR, embedding_size)    
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    count_not_present = 0
    for word, idx in word_index.items():    
        if idx < vocab_size:
            word_embedding = glove_dict.get(word)
            if word_embedding is not None:
                embedding_matrix[idx] = word_embedding
            else:
                count_not_present += 1
                embedding_matrix[idx] = np.random.randn(embedding_size)
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}, with {count_not_present} not present")
    return embedding_matrix

def load_bioword_embeddings(filename):
    return KeyedVectors.load_word2vec_format(filename, binary=True)

def load_bioword_matrix(filename, vocab_size, word_index):
    bioword_model = load_bioword_embeddings(filename)
    embedding_size = 200
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    
    count_not_present = 0
    for word, idx in word_index.items():
        if idx < vocab_size:
            if word not in bioword_model:    
                count_not_present += 1
                embedding_matrix[idx] = np.random.randn(embedding_size)
            else:
                embedding_matrix[idx] = bioword_model['word']
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}, with {count_not_present} not present")
    return embedding_matrix

def preprocess_training_data(x_train, y_train, x_val, y_val, max_words, max_sentence_length, seed=42):
    tokenizer = Tokenizer(
        num_words=max_words, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
    tokenizer.fit_on_texts(x_train)

    train_sequences = tokenizer.texts_to_sequences(x_train)
    validation_sequences = tokenizer.texts_to_sequences(x_val)

    # Pad the sequences based on the input parameter
    word_index = tokenizer.word_index
    train_data = pad_sequences(train_sequences,
                               maxlen=max_sentence_length,
                               padding='pre')
    validation_data = pad_sequences(validation_sequences,
                                    maxlen=max_sentence_length,
                                    padding='pre')

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    print('Training Data Vector: ', train_data.shape)
    print('Validation Data Vector: ', validation_data.shape)
    return (train_data, y_train), (validation_data, y_val), word_index, tokenizer