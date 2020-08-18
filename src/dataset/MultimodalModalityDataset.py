from __future__ import print_function
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import torch
import random
import pandas as pd
import skimage.io as io
import os
import numpy as np

import re
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


class MultimodalModalityDataset(torch.utils.data.Dataset):
    """
    Dataset for modality classification
    params:
        root_dir: path to the images organized in class folders
        csv_path: path to csv with image names, modality and captions
        set_keys: keys belonging to the training or validation set
    """
    
    def __init__(self, root_dir, csv_fpath, set_keys, classes, tokenizer=None, max_seq_len=200, max_words=20000, transform=None):        
        self.root_dir = root_dir
        self.transform = transform        
        
        # from the CSV, select the subset for train or val
        df_all = pd.read_csv(csv_fpath)
        
        # temp fix
        # add two samples to GLPI
        glpi_unique_sample1 = df_all[df_all['CLASS']=='GPLI'].copy()
        glpi_unique_sample1['ID'] = 9998
        glpi_unique_sample2 = df_all[df_all['CLASS']=='GPLI'].copy()
        glpi_unique_sample2['ID'] = 99989
        df_all = df_all.append(glpi_unique_sample1).append(glpi_unique_sample2)
        #######################################
        
        self.dataframe = df_all[df_all['ID'].isin(set_keys)]
        self.dataframe['CLEAN'] = self.dataframe['CAPTION'].apply(clean_str)
        self.dataframe = self.dataframe.reset_index()
        
        # tokenize
        captions = self.dataframe['CLEAN'].tolist()
        if tokenizer is None:
            self.tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
            self.tokenizer.fit_on_texts(captions)
        else:
            self.tokenizer = tokenizer
        
        caption_seqs = self.tokenizer.texts_to_sequences(captions)
        self.caption_seqs_padded = pad_sequences(caption_seqs, maxlen=max_seq_len, padding='pre')
        
        # one hot encoder for the chart type
        self.codec = LabelEncoder()
        unique_labels = classes
        unique_labels.sort()
        self.codec.fit(unique_labels)        
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # don't use idx as index but the index at ID=idx
        caption_idx = self.dataframe[self.dataframe['ID'] == idx].index[0]
        image, img_name = self.read_image(caption_idx)
        caption_seq = self.caption_seqs_padded[caption_idx]
        
        if self.transform:
            image = self.transform(image)
            caption_seq = torch.LongTensor(caption_seq)
        
        label = self.codec.transform([self.dataframe.iloc[caption_idx, 2]])
        
        return (image, label[0], caption_seq, img_name)
    
    def read_image(self, idx):
        img_id = self.dataframe.iloc[idx, 1]
        img_clef_class = self.dataframe.iloc[idx, 2]
        img_name = os.path.join(self.root_dir, img_clef_class, str(img_id) + '.jpg')
        image = io.imread(img_name)[:,:,:3]
        return image, img_name
