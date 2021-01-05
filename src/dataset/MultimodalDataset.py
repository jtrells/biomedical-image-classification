import torch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from skimage import io
from skimage.color import gray2rgb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def fit_label_encoder(unique_labels):
    le = LabelEncoder()    
    unique_labels.sort()
    le.fit(unique_labels)
    return le

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_data_path,
        base_img_dir,
        data_set,
        image_transform=None,
        tokenizer=None,
        label_name='MODALITY',
        max_input_length=300,
        target_class_col='SET',
        caption_col='CAPTION',
        path_col='PATH'
    ):
        self.base_dir = Path(base_img_dir)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.max_input_length = max_input_length
        self.path_col = path_col
        self.caption_col = caption_col

        # filter the train, val or test data values
        self.df = pd.read_csv(csv_data_path, sep='\t')
        if type(data_set) == str:
            self.df = self.df[self.df[target_class_col] == data_set]
        else:
            self.df = self.df[self.df['ID'].isin(data_set)]
        self.le = fit_label_encoder(self.df[label_name].unique().tolist())
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()                
        
        image = self.read_image(idx)
        caption = self.df.iloc[idx][self.caption_col]
        modality = self.df.iloc[idx][self.label_name]
        label = self.le.transform([modality])

        if self.image_transform:
            image = self.image_transform(image)
        if self.tokenizer:
            seqs = self.tokenizer.texts_to_sequences([caption])
            padded_seqs = pad_sequences(seqs, maxlen=self.max_input_length, padding='pre')
            padded_seqs = torch.LongTensor(padded_seqs)
#             padded_seqs = torch.Tensor(padded_seqs)
#             padded_seqs = padded_seqs.view(padded_seqs.size(0), -1)
        
        return (padded_seqs, image, label[0])
    
    def read_image(self, idx):
        img_path = self.base_dir / self.df.iloc[idx][self.path_col]
        image = io.imread(img_path)
        # some clef images were grayscale
        if len(image.shape) == 2:
            image = gray2rgb(image)
        else:
            image = image[:, :, :3]
        return image