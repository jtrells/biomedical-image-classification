from __future__ import print_function
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import torch
import random
import pandas as pd
import skimage
import os
import numpy as np

class ModalityDataset(torch.utils.data.Dataset):
    """
    Dataset for modality classification
    params:
        root_dir: path to the images organized in class folders
        csv_path: path to csv with the image, modality, high_modality labels
        set_keys: keys belonging to the training or validation set
    """
    
    def __init__(self, root_dir, csv_fpath, set_keys, classes, transform=None):        
        self.root_dir = root_dir
        self.transform = transform        
        
        # from the CSV, select the subset for train or val
        df_all = pd.read_csv(csv_fpath)
        self.dataframe = df_all[df_all['ID'].isin(set_keys)]
        
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
        
        image, img_name = self.read_image(idx)
        if self.transform:
            image = self.transform(image)
        
        label = self.codec.transform([self.dataframe.iloc[idx, 3]])
        return (image, label[0], img_name)
    
    def read_image(self, idx):
        img_id = self.dataframe.iloc[idx, 1]
        img_clef_class = self.dataframe.iloc[idx, 2]
        img_name = os.path.join(self.root_dir, img_clef_class, str(img_id) + '.jpg')
        image = skimage.io.imread(img_name)[:,:,:3]
        return image, img_name
