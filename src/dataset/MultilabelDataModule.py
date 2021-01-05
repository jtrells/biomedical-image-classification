import pytorch_lightning as pl
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from .MultilabelDataset import MultilabelDataset
from torch.utils.data import DataLoader

class MultilabelDataModule(pl.LightningDataModule):
    '''
    Prepare train, validation and test datasets for 
    multilabel classification with two approaches:
    1. Pass the train, validation and test partitions
       in the 'SET' column. Useful for hold-out cases.
    2. Pass the kfold values, take the current k as
       the hold-out set and return the rest as training.
       For test, use the kfold value -1.
    '''
    def __init__(self,
                 batch_size,
                 data_path,
                 vocab_size,
                 max_input_length,
                 num_workers=8,
                 random_state=443,
                 test_size=0.2, # replaced by k_fold when using k_fold splitting
                 kfold_col=None,
                 preprocess_fn=None):
        super().__init__()
        
        self.use_kfold = kfold_col is not None
        self.kfold_col = kfold_col
        
        self.batch_size = batch_size
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_input_length = max_input_length
        self.tokenizer = None
        self.num_workers = num_workers
        self.random_state = random_state
        self.test_size = test_size
        self.preprocess_fn = preprocess_fn

    def prepare_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t')
        self.caption_col = 'CAPTION'
        if self.preprocess_fn is not None:
            self.caption_col = 'PR_CAPTION'
            self.df.loc[:, self.caption_col] = self.df.apply(lambda x: self.preprocess_fn(x['CAPTION']), axis=1)
        
        if self.use_kfold:
            self.n_folds = len(self.df[self.kfold_col].unique()) - 1 # don't count test
            self.df_test = self.df[self.df[self.kfold_col] == -1].reset_index(drop=True)
        else:
            self.df_test = self.df[self.df['SET']=='TEST'].reset_index(drop=True)
        
    def setup(self, k_fold_idx=None):                
        if self.use_kfold:
            if k_fold_idx == None or k_fold_idx >= self.n_folds:
                raise Exception(f"k_fold_idx needs to an integer between 0 and {self.n_folds-1}")
            df_not_test  = self.df[self.df[self.kfold_col] != -1].reset_index(drop=True)            
            
            self.df_train = df_not_test[df_not_test[self.kfold_col] != k_fold_idx].reset_index(drop=True)
            self.df_valid = df_not_test[df_not_test[self.kfold_col] == k_fold_idx].reset_index(drop=True)        
        else:
            df_not_set   = self.df[self.df['SET']=='TRAIN']
            ids = np.arange(df_not_set.shape[0])
            
            if self.test_size > 0:                
                train_idx, valid_idx = train_test_split(ids, test_size=self.test_size, random_state=self.random_state)
            else:
                train_idx = ids
                valid_idx = ids                
            self.df_train = df_not_set[train_idx]
            self.df_valid = df_not_set[valid_idx]
            
        captions_train = self.df_train[self.caption_col].values        
                        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
        self.tokenizer.fit_on_texts(captions_train)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1          
        
    def train_dataloader(self):        
        dataset = MultilabelDataset(
            self.df_train,
            tokenizer=self.tokenizer,
            columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
            max_input_length=self.max_input_length,
            caption_col=self.caption_col
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):        
        dataset = MultilabelDataset(
            self.df_valid,
            tokenizer=self.tokenizer,
            columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
            max_input_length=self.max_input_length,
            caption_col=self.caption_col
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)      
    
    def test_dataloader(self):        
        dataset = MultilabelDataset(
            self.df_test,
            tokenizer=self.tokenizer,
            columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
            max_input_length=self.max_input_length,
            caption_col=self.caption_col
        )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)     