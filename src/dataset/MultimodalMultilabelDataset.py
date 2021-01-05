import torch
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from skimage import io
from skimage.color import gray2rgb

def fit_label_encoder(unique_labels):
    le = LabelEncoder()    
    unique_labels.sort()
    le.fit(unique_labels)
    return le

class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_data_path,
                 base_img_dir,
                 idx_list,
                 tokenizer=None,
                 image_transform=None,
                 columns=['DMEL', 'DMFL', 'DMLI', 'DMTR'],
                 label_name='MODALITY',
                 max_input_length=300):
        self.base_dir = Path(base_img_dir)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.columns   = columns
        self.label_name = label_name
        self.max_input_length = max_input_length
        
        self.df = pd.read_csv(csv_data_path, sep='\t')
        self.df = self.df[self.df['ID'].isin(idx_list)]
        self.le = fit_label_encoder(self.df['MODALITY'].unique().tolist())
    
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

    def read_image(self, idx):
        img_path = self.base_dir / self.df.iloc[idx]['PATH']
        image = io.imread(img_path)
        # some clef images were grayscale
        if len(image.shape) == 2:
            image = gray2rgb(image)
        else:
            image = image[:, :, :3]
        return image