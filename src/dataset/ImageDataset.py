import torch
import pandas as pd
from pathlib import Path
from skimage import io
from skimage.color import gray2rgb


class ImageDataset(torch.utils.data.Dataset): 
    def __init__(
        self,
        csv_data_path,
        label_encoder,
        base_img_dir,
        data_set,
        image_transform=None,
        label_name='MODALITY',
        target_class_col='SET',
        path_col='PATH'
    ):
        self.base_dir = Path(base_img_dir)
        self.image_transform = image_transform
        self.label_name = label_name
        self.path_col = path_col
        self.le = label_encoder

        # filter the train, val or test data values
        self.df = pd.read_csv(csv_data_path, sep='\t')
        if type(data_set) == str:
            self.df = self.df[self.df[target_class_col] == data_set]
        else:
            self.df = self.df[self.df['ID'].isin(data_set)]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()                
        
        image = self.read_image(idx)
        modality = self.df.iloc[idx][self.label_name]
        label = self.le.transform([modality])

        if self.image_transform:
            image = self.image_transform(image)
        return (image, label[0])
    
    def read_image(self, idx):
        img_path = self.base_dir / self.df.iloc[idx][self.path_col]
        image = io.imread(img_path)
        # some clef images were grayscale
        if len(image.shape) == 2:
            image = gray2rgb(image)
        else:
            image = image[:, :, :3]
        return image
