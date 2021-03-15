import torch
from tqdm.notebook import tqdm
import numpy as np


def get_vector_representation(data_loader, model, device):
    model.to(device)
    # Put the model in eval mode
    model.eval()
    # List for store final predictions
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            predictions = model(data[0])
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return np.vstack((final_predictions))[:,:,0,0]
