import torch
import pandas as pd
from skimage import io
from skimage.color import gray2rgb
from sklearn.preprocessing import LabelEncoder


class MicroscopyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_set, transform=None):
        self.transform = transform
        self.image_set = image_set

        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['SET'] == self.image_set]

        # one hot encoder for classes
        self.codec = LabelEncoder()
        unique_labels = self.df['MODALITY'].unique().tolist()
        unique_labels.sort()
        self.codec.fit(unique_labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # is this idx the row? or identifier?, loader shuffles, not df.
        # df.index[df['ID']==4][0]
        image = self.read_image(idx)
        img_name = self.df.iloc[idx, 1]
        modality = self.df.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        label = self.codec.transform([modality])
        return (image, label[0], img_name)

    def read_image(self, idx):
        img_path = self.df.iloc[idx, 4]
        image = io.imread(img_path)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        else:
            image = image[:, :, :3]
        return image
