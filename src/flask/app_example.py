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
from dataset.ImageDataset import ImageDataset
from models.EfficientNetClass import EfficientNetClass
from models.ResNetClass import ResNetClass
from torchvision import transforms
from flask import Flask
# -

root_path = '/berrios-3'
PORT = 6007

# +
app = Flask(__name__)

@app.route(root_path)
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = PORT)
