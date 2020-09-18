# TODO: Not using clear_str now, just basic preprocessing from tokenizer.

import pytorch_lightning as pl
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from .MultimodalDataset import MultimodalDataset

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


class CaptionDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 batch_size,
                 data_path,
                 vocab_size,
                 max_input_length,
                 base_img_dir,
                 num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_input_length = max_input_length
        self.base_img_dir = base_img_dir
        self.le = LabelEncoder()
        self.tokenizer = None
        self.num_workers = num_workers
    
    def prepare_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t')
        #TODO check if I need to add the clean_str
        
    def setup(self):
        train_df = self.df[self.df['SET']=='TRAIN']                
        x0_train= train_df['CAPTION'].values
                        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
        self.tokenizer.fit_on_texts(x0_train)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1                 
    
    def train_dataloader(self):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ]
        image_transform = transforms.Compose(transform_list)
        
        dataset = MultimodalDataset(
            self.data_path,
            self.base_img_dir,
            'TRAIN',
            image_transform=image_transform,
            tokenizer=self.tokenizer,
            label_name='MODALITY',
            max_input_length=self.max_input_length
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        image_transform = transforms.Compose(transform_list)
        
        dataset = MultimodalDataset(
            self.data_path,
            self.base_img_dir,
            'VAL',
            image_transform=image_transform,
            tokenizer=self.tokenizer,
            label_name='MODALITY',
            max_input_length=self.max_input_length
        )        
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        image_transform = transforms.Compose(transform_list)
        
        dataset = MultimodalDataset(
            self.data_path,
            self.base_img_dir,
            'TEST',
            image_transform=image_transform,
            tokenizer=self.tokenizer,
            label_name='MODALITY',
            max_input_length=self.max_input_length
        )        
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)