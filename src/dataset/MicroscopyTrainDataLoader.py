# +
import random
from torchvision import transforms
from .MicroscopyDataset import MicroscopyDataset


class MicroscopyTrainDataLoader():
    def __init__(self, csv_path, seed=443):
        self.csv_path = csv_path

    def get_train_dataset(self, normalized=True):
        return self._get_dataset('train', normalized=normalized)

    def get_val_dataset(self, normalized=True):
        return self._get_dataset('validation', normalized=normalized)

    def _get_dataset(self, image_set, normalized=True):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

        if normalized:
            transform_list.append(transforms.Normalize(
                [0.4857, 0.4740, 0.4755], [0.3648, 0.3557, 0.3669]))

        transform = transforms.Compose(transform_list)
        return MicroscopyDataset(self.csv_path, image_set=image_set, transform=transform)
