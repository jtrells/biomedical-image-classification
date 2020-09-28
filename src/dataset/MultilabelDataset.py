import torch
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_data_path,
                 idx_list,
                 tokenizer=None,
                 columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
                 max_input_length=300):
        self.tokenizer = tokenizer
        self.columns   = columns
        self.max_input_length = max_input_length
        
        self.df = pd.read_csv(csv_data_path, sep='\t')
        self.df = self.df[self.df['ID'].isin(idx_list)]
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        caption = self.df.iloc[idx]['CAPTION']
        modalities = self.df.iloc[idx][self.columns].values.astype(int)
        if self.tokenizer:
            seqs = self.tokenizer.texts_to_sequences([caption])
            padded_seqs = pad_sequences(seqs, maxlen=self.max_input_length, padding='pre')
            padded_seqs = torch.LongTensor(padded_seqs)
        
        # second parameter is the expected image output for a multimodal model
        return (padded_seqs, 0, torch.LongTensor(modalities))
               