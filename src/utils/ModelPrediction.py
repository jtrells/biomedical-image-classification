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


def run_metrics(df,y_true,y_pred,it):
    acc             = accuracy_score(y_true, y_pred)
    balanced_acc    = balanced_accuracy_score(y_true, y_pred)
    macro_f1        = f1_score (y_true,y_pred,average='macro')
    macro_recall    = recall_score(y_true,y_pred,average='macro')
    macro_precision = precision_score(y_true,y_pred, average='macro')
    df = pd.concat([df,pd.DataFrame({'Iteration':[it],'Acc':[acc],'Macro - F1':[macro_f1],
                                      'Macro - Recall':[macro_recall],'Macro_Precision':[macro_precision]})],axis = 1)
