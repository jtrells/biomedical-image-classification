import pytorch_lightning as pl
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from .MultilabelDataset import MultilabelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

class MultilabelDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 data_path,
                 vocab_size,
                 max_input_length,
                 num_workers=8,
                 seed=443,
                 test_size=0.2, # replaced by k_fold when using k_fold splitting
                 k_fold=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_input_length = max_input_length
        self.tokenizer = None
        self.num_workers = num_workers
        self.seed = seed
        self.test_size = test_size
        self.k_fold = k_fold

    def prepare_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t')
        
        if self.k_fold:
            train_df = self.df[self.df['SET']=='TRAIN']
            ids = train_df['ID'].values
            
            self.kf = KFold(n_splits=self.k_fold, random_state=self.seed, shuffle=True)
        
    def setup(self, k_fold_idx=None):
        train_df = self.df[self.df['SET']=='TRAIN']
        if self.k_fold is None:
            ids = train_df['ID'].values
            if self.test_size > 0:                
                self.train_idx, self.valid_idx = train_test_split(ids,
                                test_size=self.test_size,
                                random_state=self.seed)
            else:
                self.train_idx = ids
                self.valid_idx = ids
        else:
            if k_fold_idx is not None:
                i = 0
                for train_idx, val_idx in self.kf.split(train_df['CAPTION'].values):
                    if i == k_fold_idx - 1: break
                    else:
                        self.train_idx = train_idx
                        self.valid_idx = val_idx
                        i += 1
            else:
                raise Exception("The k-fold index cannot be null when using k-fold splitting")       
        
        # now filter train data frame based on the partition
        train_df = train_df[train_df['ID'].isin(self.train_idx)]
        x0_train= train_df['CAPTION'].values
                        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
        self.tokenizer.fit_on_texts(x0_train)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1          
        
    def train_dataloader(self):        
        dataset = MultilabelDataset(
            self.data_path,
            self.train_idx,
            tokenizer=self.tokenizer,
            columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
            max_input_length=self.max_input_length
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):        
        dataset = MultilabelDataset(
            self.data_path,
            self.valid_idx,
            tokenizer=self.tokenizer,
            columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
            max_input_length=self.max_input_length
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)      
    
    def test_dataloader(self):        
        return None      