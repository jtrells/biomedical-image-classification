import os
import sys
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
# 1. General Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
# 2. DataModule & Class Libraries
from utils.label_encoder import label_encoder_target
from dataset.ImageDataModule import ImageDataModule
from dataset.ImageDataset import ImageDataset,EvalImageDataset
from models.EfficientNetClass import EfficientNetClass
from models.ResNetClass import ResNetClass
from torchvision import transforms
from utils.ModelPrediction import get_prediction,all_pred
from flask import Flask
from flask import request
from flask import render_template
import torch
from flask import jsonify


# +
ROOT_PATH = '/berrios-3'
API       = 'api'
PORT      = 6007
MODEL_NAME = '/mnt/artifacts/experiments/Biomedical-Image-Classification-Higher-Modality/v5hru53s/iconic-sweep-1/final.pt'

DATA_PATH  = '../../data/higher_modality_vol1.csv'
LABEL_ENCODER,DICT_LABEL = label_encoder_target(pd.read_csv(DATA_PATH, sep='\t'),target_col='higher_modality')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# +
def predict_image(model_path,image_location,le_encoder):
    resnet_model = ResNetClass.load_from_checkpoint(model_path)
    
    test_aug = [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(resnet_model.hparams['mean_dataset'],resnet_model.hparams['std_dataset'])
            ]
    test_aug = transforms.Compose(test_aug)
    
    df = pd.DataFrame({'img_path':[image_location]})
    
    test_dataset   = EvalImageDataset(df,
                                      image_transform=test_aug,
                                      path_col='img_path')
    del df
    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size = 32,
                        shuffle = False,
                        num_workers = 72
                      )
    pred = get_prediction(test_dataloader,resnet_model.to(DEVICE), DEVICE)
    
    return le_encoder.inverse_transform(pred)

def predict_v2(model_path,image_location,le_encoder):
    resnet_model = ResNetClass.load_from_checkpoint(model_path)
    
    test_aug = [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(resnet_model.hparams['mean_dataset'],resnet_model.hparams['std_dataset'])
            ]
    test_aug = transforms.Compose(test_aug)
    
    df = pd.DataFrame({'img_path':[image_location]})
    
    test_dataset   = EvalImageDataset(df,
                                      image_transform=test_aug,
                                      path_col='img_path')
    del df
    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size = 32,
                        shuffle = False,
                        num_workers = 72
                      )
    final_all_probs,final_predictions_indx,final_probs = all_pred(test_dataloader,resnet_model.to(DEVICE), DEVICE)
    
    return final_all_probs,final_probs,le_encoder.inverse_transform(final_predictions_indx),final_predictions_indx

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



# +
app = Flask(__name__)

# Basic functionality to check if server is working
@app.route(f'{ROOT_PATH}')
def check():
    return "Api is working!"

@app.route(f"{os.path.join(ROOT_PATH,API,'predict_try')}",methods=['POST'])
def predict_try():
    '''
    the post would require a path in order to make the prediction
    '''
    if request.method == 'POST':
        request_data = request.get_json()
        model_path = request_data['ModelPath']
        img_path   = request_data['ImgPath']
        print(f'Model Path is {model_path}')
        print(f'Image path is {img_path}')
        prediction = predict_image(model_path,img_path,LABEL_ENCODER)[0]
        x = {'Image':img_path,'prediction':prediction}
        return jsonify(x)

    
@app.route(f"{os.path.join(ROOT_PATH,API,'predict_json')}",methods=['POST'])
def predict_json():
    '''
    the post would require a path in order to make the prediction
    '''
    if request.method == 'POST':
        request_data = request.get_json()
        model_path = request_data['ModelPath']
        img_path   = request_data['ImgPath']
        print(f'Model Path is {model_path}')
        print(f'Image path is {img_path}')
        final_all_probs,final_probs,label_predicted,final_predictions_indx = predict_v2(model_path,img_path,LABEL_ENCODER)
        print( f'''
        All probabilities: {final_all_probs[0]}
        Final probs: {final_probs[0]}
        label predicted: {label_predicted[0]}
        idx : {final_predictions_indx[0]}
        ''')
        x = {'Image':img_path,'all_probs':final_all_probs[0],'label_predicted':label_predicted[0],'prob_class':final_probs[0],'idx_highest':final_predictions_indx[0]}
        #x = {'Image':img_path,'idx_highest':final_predictions_indx[0]}
        return json.dumps(x, cls=NpEncoder)



# -

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = PORT)
