import torch
from torch.utils.data import DataLoader

def calc_dataset_mean_std(train_tensor_dataset, batch_size=32, num_workers=0):
    """
    Args:
        train_tensor_dataset: dataset with just a ToTensor transformation
          e.g. datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
    """    
    loader = DataLoader(train_tensor_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    mean = 0.0
    for images, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)
    
    var = 0.0
    sample_img, _, _ = train_tensor_dataset[0]
    # for std calculations
    h, w = sample_img.shape[1], sample_img.shape[2]
    
    for images, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*w*h))
    
    return mean, std