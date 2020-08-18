# +
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from .ModalityDataset import ModalityDataset
from .MultimodalModalityDataset import MultimodalModalityDataset
from tensorflow.keras.preprocessing.text import Tokenizer

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


class TrainDataLoader():
    def __init__(self, train_img_dir, csv_path, classes, variant="multimodal", seed=443, val_size=0.20, max_words = 20000, max_seq_len=200):
        # train hosts training and validation images
        self.variant = variant
        if self.variant == "high_level":
            self.y_name = "HIGH_MODALITY"
        else:
            self.y_name = "CLASS"
        
        self.train_img_dir = train_img_dir
        self.csv_path = csv_path        
        self.classes = classes
        self.max_seq_len = max_seq_len
        self.max_words = max_words
        self.train_keys, self.val_keys = self._create_validation_set(val_size)
        
        # fit a text tokenizer for multimodal variant
        if variant == "multimodal":
            df_all = pd.read_csv(csv_path)
            df_all = df_all[df_all['ID'].isin(self.train_keys)]
            df_all['CLEAN'] = df_all['CAPTION'].apply(clean_str)
            
            captions = df_all['CLEAN'].tolist()
            self.tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
            self.tokenizer.fit_on_texts(captions)
        
    def _create_validation_set(self, val_size, seed=443):
        ''' get the keys of items that belong to the training and validation 
            sets in a stratified manner '''
        random.seed(seed)
        np.random.seed(seed)
        
        data_df = pd.read_csv(self.csv_path)
        # add two samples to GLPI
        glpi_unique_sample1 = data_df[data_df['CLASS']=='GPLI'].copy()
        glpi_unique_sample1['ID'] = 9998
        glpi_unique_sample2 = data_df[data_df['CLASS']=='GPLI'].copy()
        glpi_unique_sample2['ID'] = 99989
        data_df = data_df.append(glpi_unique_sample1).append(glpi_unique_sample2)
        #######################################
        labels_dict = data_df.set_index('ID').T.to_dict('list')
        
        X = list(labels_dict.keys())
        col_idx = data_df.columns.get_loc(self.y_name)
        y = [labels_dict[x][col_idx] for x in X] # HIGH_MODALITY in position 2
        X = np.array(X)
        # not sure if we require to shuffle...
        train_keys, val_keys, _, _ = train_test_split(X, y, stratify=y, test_size=val_size, shuffle=True)
        print("There are {0} training images and {1} validation images".format(len(train_keys), len(val_keys)))
        
        return train_keys, val_keys
    
    def get_train_dataset(self, normalized=True):        
        return self._get_dataset(self.train_keys, normalized=normalized)
    
    def get_val_dataset(self, normalized=True):        
        return self._get_dataset(self.val_keys, normalized=normalized)

    def _get_dataset(self, keys, normalized=True):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        
        if normalized:
            transform_list.append(transforms.Normalize([0.7364, 0.7319, 0.7295], [0.3538, 0.3543, 0.3593]))
            
        transform = transforms.Compose(transform_list)
        if self.variant == "high_level":
            return ModalityDataset(self.train_img_dir, self.csv_path, keys, self.classes, transform=transform)
        elif self.variant == "multimodal":
            return MultimodalModalityDataset(self.train_img_dir, self.csv_path, keys, self.classes, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, max_words=self.max_words, transform=transform)
