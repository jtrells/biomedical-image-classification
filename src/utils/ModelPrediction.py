import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd


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


def get_probs(data_loader, model, device,loss_with = 'logits'):
    # Put the model in eval mode
    model.eval()
    m = nn.Softmax(dim=1)
    # List for store final predictions
    final_probs = []
    if loss_with == 'logits':
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                data  = data.to(device)
                probs = model(data)
                probs = m(probs)
                final_probs.append(probs.cpu())
        return np.vstack(final_probs)
    else:
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                data  = data.to(device)
                probs = model(data)
                final_probs.append(probs.cpu())
        return np.vstack(final_probs)


def all_pred(data_loader,model,device):
    # Put the model in eval mode
    model.eval()
    # List for store final predictions
    final_predictions = []
    final_probs       = []
    final_all_probs   = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            data = data.to(device)
            probs = model(data)
            max_probs, predictions = torch.max(probs, dim=1)
            predictions = predictions.cpu()
            max_probs   = max_probs.cpu()
            probs       = probs.cpu()
            final_predictions.append(predictions)
            final_probs.append(max_probs)
            final_all_probs.append(probs)
    return np.hstack(final_all_probs),np.hstack(final_predictions),np.hstack(final_probs)


def run_metrics(y_true,y_pred):
    acc             = np.round(100*accuracy_score(y_true, y_pred),2)
    balanced_acc    = np.round(100*balanced_accuracy_score(y_true, y_pred),2)
    macro_f1        = np.round(100*f1_score (y_true,y_pred,average='macro'),2)
    macro_recall    = np.round(100*recall_score(y_true,y_pred,average='macro'),2)
    macro_precision = np.round(100*precision_score(y_true,y_pred, average='macro'),2)
    return acc,balanced_acc,macro_f1,macro_recall,macro_precision
