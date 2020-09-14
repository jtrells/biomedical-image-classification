# +
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from .ModalityDataset import ModalityDataset

class TrainDataLoader():
    def __init__(self, train_img_dir, csv_path, classes, seed=443, val_size=0.20):
        # train hosts training and validation images
        self.train_img_dir = train_img_dir
        self.csv_path = csv_path        
        self.classes = classes
        self.train_keys, self.val_keys = self._create_validation_set(val_size)
        
    def _create_validation_set(self, val_size, seed=443):
        ''' get the keys of items that belong to the training and validation 
            sets in a stratified manner '''
        random.seed(seed)
        np.random.seed(seed)
        
        data_df = pd.read_csv(self.csv_path)
        labels_dict = data_df.set_index('ID').T.to_dict('list')
        
        X = list(labels_dict.keys())
        y = [labels_dict[x][2] for x in X] # HIGH_MODALITY in position 2
        X = np.array(X)        
        train_keys, val_keys, _, _ = train_test_split(X, y, stratify=y, test_size=val_size)
        
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
        return ModalityDataset(self.train_img_dir, self.csv_path, keys, self.classes, transform=transform)
