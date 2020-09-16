import pytorch_lightning as pl
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

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
    
    def __init__(self, batch_size, data_path, vocab_size, max_input_length):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.max_input_length = max_input_length
        self.le = LabelEncoder()
    
    def prepare_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t')
        #TODO check if I need to add the clean_str
        
    def setup(self):
        train_df = self.df[self.df['SET']=='TRAIN']
        val_df = self.df[self.df['SET']=='VAL']
        test_df = self.df[self.df['SET']=='TEST']
        
        x0_train, y0_train = train_df['CAPTION'].values, train_df['MODALITY'].values
        x0_val, y0_val = val_df['CAPTION'].values, val_df['MODALITY'].values
        x0_test, y0_test = test_df['CAPTION'].values, test_df['MODALITY'].values
                
        tokenizer = Tokenizer(num_words=self.vocab_size, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n\'')
        tokenizer.fit_on_texts(x0_train)
        self.word_index = tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1        
        self.le.fit(y0_train)
        
        train_seqs = tokenizer.texts_to_sequences(x0_train)
        val_seqs = tokenizer.texts_to_sequences(x0_val)
        test_seqs = tokenizer.texts_to_sequences(x0_test)
                
        X_train = pad_sequences(train_seqs, maxlen=self.max_input_length, padding='pre')
        X_val   = pad_sequences(val_seqs,   maxlen=self.max_input_length, padding='pre')
        X_test  = pad_sequences(test_seqs,  maxlen=self.max_input_length, padding='pre')                
    
        
        self.captions_train = TensorDataset(torch.LongTensor(X_train),
                                                 torch.LongTensor(self.le.transform(y0_train)))
        self.captions_val   = TensorDataset(torch.LongTensor(X_val),
                                                 torch.LongTensor(self.le.transform(y0_val)))
        self.captions_test  = TensorDataset(torch.LongTensor(X_test),
                                                 torch.LongTensor(self.le.transform(y0_test)))          
    
    def train_dataloader(self):
        return DataLoader(self.captions_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.captions_val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.captions_test, batch_size=self.batch_size, shuffle=False)