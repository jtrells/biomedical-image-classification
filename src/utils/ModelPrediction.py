import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np


def get_prediction(data_loader, model, device):
    # Put the model in eval mode
    model.eval()
    # List for store final predictions
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            data = data.to(device)
            predictions = model(data)
            #Get class prediction
            _,predictions = torch.max(predictions, dim=1)
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return np.hstack(final_predictions)
