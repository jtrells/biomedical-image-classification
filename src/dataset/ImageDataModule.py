import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from .ImageDataset import ImageDataset
from sklearn.utils import class_weight
from pytorch_lightning import Trainer, seed_everything
from pathlib import Path
from ..utils.datasets import remove_small_classes

class ImageDataModule(pl.LightningDataModule):
    def __init__(self,
                 label_encoder,
                 batch_size,
                 data_path,
                 base_img_dir,
                 seed=42,
                 image_transforms=[],
                 num_workers=8,
                 target_class_col='SET',
                 caption_col='CAPTION',
                 modality_col='MODALITY',
                 path_col='PATH',
                 shuffle_train=True):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.base_img_dir = base_img_dir
        self.le = label_encoder
        self.num_workers = num_workers
        self.target_class_col = target_class_col
        self.modality_col = modality_col
        self.path_col = path_col
        self.seed = seed
        self.image_transforms_train = image_transforms[0]
        self.image_transforms_valid = image_transforms[1]
        self.image_transforms_test = image_transforms[2]
        self.shuffle_train = shuffle_train

    def prepare_data(self):
        path = Path(self.data_path)
        if path.suffix.lower() == '.csv':
            self.df = pd.read_csv(self.data_path, sep='\t')
        else:
            self.df = pd.read_parquet(self.data_path)
            self.df = remove_small_classes(self.df, self.modality_col, threshold=100)
        # Always run the prepare data in order to get the same results in training

    def set_seed(self):
        seed_everything(self.seed)

    def setup(self):
        train_df = self.df[self.df[self.target_class_col] == 'TRAIN']
        y_train = train_df[self.modality_col].values
        # calculate a class weight vector

        # self.class_weights = class_weight.compute_class_weight('balanced', classes=self.le.classes_, y=y_train)
        self.class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        # better remove samples if there are less than 50
        # if set(np.unique(y_train)) != set(self.le.classes_):
        #     # special cases when users defined a class but we don't have training data
        #     # e.g. electron other had one sample in validation and one in test
        #     weights = []
        #     weights_dictionary = dict(zip(np.unique(y_train), weights))
        #     for class_label in self.le.classes_:
        #         if class_label in weights_dictionary:
        #             weights.append(weights_dictionary[class_label])
        #         else:
        #             weights.append(0.0)
        #     self.class_weights = weights

        del train_df, y_train

    def train_dataloader(self):

        train_dataset = ImageDataset(
            csv_data_path=self.data_path,
            label_encoder=self.le,
            base_img_dir=self.base_img_dir,
            data_set='TRAIN',
            image_transform=self.image_transforms_train,
            label_name=self.modality_col,
            target_class_col=self.target_class_col,
            path_col=self.path_col)

        return DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle_train,
                          num_workers=self.num_workers)

    def val_dataloader(self):

        val_dataset = ImageDataset(
            csv_data_path=self.data_path,
            label_encoder=self.le,
            base_img_dir=self.base_img_dir,
            data_set='VAL',
            image_transform=self.image_transforms_valid,
            label_name=self.modality_col,
            target_class_col=self.target_class_col,
            path_col=self.path_col)

        return DataLoader(dataset=val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):

        test_dataset = ImageDataset(
            csv_data_path=self.data_path,
            label_encoder=self.le,
            base_img_dir=self.base_img_dir,
            data_set='TEST',
            image_transform=self.image_transforms_valid,
            label_name=self.modality_col,
            target_class_col=self.target_class_col,
            path_col=self.path_col)

        return DataLoader(dataset=test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
