# +
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
from utils.ModelPrediction import get_prediction
from flask import Flask
from flask import request
from flask import render_template
import torch

# +
# General Variables
ROOT_PATH  = '/berrios-3'
PORT       = 6007
UPLOAD_FOLDER = '../../data/images_loaded/'
MODEL_NAME = '/mnt/artifacts/experiments/Biomedical-Image-Classification-Higher-Modality/v5hru53s/iconic-sweep-1/final.pt'

DATA_PATH  = '../../data/higher_modality_vol1.csv'
LABEL_ENCODER,DICT_LABEL = label_encoder_target(pd.read_csv(DATA_PATH, sep='\t'),target_col='higher_modality')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# +
# Load the Model
resnet_model = ResNetClass.load_from_checkpoint(MODEL_NAME)
parts_resnet = [i for i in resnet_model.children()]

# Transformation
test_aug = [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(resnet_model.hparams['mean_dataset'],resnet_model.hparams['std_dataset'])
            ]
test_aug = transforms.Compose(test_aug)

def eval_image_flask(image_location,resnet_model,test_aug,le_encoder):
    df = pd.DataFrame({'img_path':[image_location]})
    test_dataset   = EvalImageDataset(df,
                                      UPLOAD_FOLDER,
                                      image_transform=test_aug,
                                      path_col='img_path')
    del df
    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size = 32,
                        shuffle = False,
                        num_workers = 72
                      )
    pred = get_prediction(test_dataloader,resnet_model.to('cuda'), 'cuda')
    
    return le_encoder.inverse_transform(pred)             


# +
#image_location = '/mnt/subfigure-classification/2016/train/DMFL/11373_2007_9226_Fig1_HTML-10.jpg'
#eval_image_flask(image_location,resnet_model,test_aug,LABEL_ENCODER)[0]

# +
app = Flask(__name__)

#@app.route(ROOT_PATH)
#def hello():
#    return "Hello World!"

@app.route(ROOT_PATH,methods = ['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = eval_image_flask(image_location,resnet_model,test_aug,LABEL_ENCODER)[0]
            print(image_file.filename)
            print(image_location)
            return render_template('index.html', Prediction = pred, image_loc = image_file.filename) 
    return render_template('index.html',Prediction = 0,image_loc = None)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = PORT)
