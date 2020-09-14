"""
Train an image classifier on the CHART-Synthetic dataset
run: CUDA_VISIBLE_DEVICES=X python train.py --dataset-path /path1 --labels-path /path2 --out=dir /path3
     where X=0 or X=1 or ... X=0,..,n depending on the GPUs available
     check args below for more parameters
last run:
CUDA_VISIBLE_DEVICES=1,2 python src/train.py --batch-size 64 --test-batch-size 64 --epochs 100 --lr 1e-3 --num-output-classes 6 --dataset-path /mnt/clef/imageclef_2016/train --labels-path /mnt/biomedical-image-classification/labels/clef16_train.csv --out-dir /mnt/models/high-level-modality --infra compass --architecture resnet50
"""
import os
import sys
print(os.getcwd())
module_path = os.path.abspath(os.path.join('../../../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

import torch.nn as nn
import argparse
from torchvision import models
from dataset.MicroscopyTrainDataLoader import MicroscopyTrainDataLoader
from ExperimentalRun import ExperimentalRun
from microscopy import experiment, get_model
import json
import torch

import torch.nn.functional as F

class MyEnsemble(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5, model6, nb_classes=4):
        super(MyEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        # Remove last linear layer
        self.model1.fc = nn.Identity()
        self.model2.fc = nn.Identity()
        self.model3.fc = nn.Identity()
        self.model4.fc = nn.Identity()
        self.model5.fc = nn.Identity()
        self.model6.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(2048+2048+2048+1024+1024+1024, nb_classes)
        
    def forward(self, x):
        x1 = self.model1(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model2(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.model3(x)
        x3 = x3.view(x3.size(0), -1)
        x4 = self.model4(x)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.model5(x)
        x5 = x5.view(x5.size(0), -1)
        x6 = self.model6(x)
        x6 = x6.view(x6.size(0), -1)
        
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

    
def load_shallow_model(model_id, model_dict):
    model_name, experiment_name = model_id.split('.')
    model = get_model(model_name, "shallow", 4, layers=model_dict[model_id]['layers'], pretrained=True)
    
    checkpoint = torch.load('/workspace/outputs/{0}/checkpoint.pt'.format(model_dict[model_id]['id']))
    model.load_state_dict(checkpoint)
    
    return model    
    
def main():
    parser = argparse.ArgumentParser(description='Microscopy Classification CLEF')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=443, metavar='S', help='random seed (default: 443)')
    parser.add_argument('--patience', type=int, default=5, metavar='N', help='how many epochs to wait before stopping a non-improving execution')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N', help='processors for data loading tasks')
    parser.add_argument('--num-output-classes', type=int, default=4, metavar='N', help='number of classes in the dataset')
    parser.add_argument('--csv-path', type=str, default='/workspace/labels/microscopy.csv', help='location of input images')
    parser.add_argument('--out-dir', type=str, default='/workspace/outputs', help='location for output data')
    parser.add_argument('--project_name', type=str, default='microscopy', help='Project name on Wandb')
    parser.add_argument('--architecture', type=str, default='resnet18', help='Highlevel description of the deep learning model used')
    parser.add_argument('--experiment', type=str, default='fc', help='Experiment variation')
    parser.add_argument('--pretrained', type=int, default='1', help='pretrain on imagenet')
    parser.add_argument('--infra', type=str, default='', help='Description of the infrastructure used for training')
    parser.add_argument('--notes', type=str, default='', help='Any particular note about the run')
    parser.add_argument('--augmentation', type=int, default='0', help='Use data augmentation')
    parser.add_argument('--weight-sampler', type=int, default='0', help='Weight sampling for unbalanced set')
    parser.add_argument('--weight-loss', type=int, default='0', help='Apply a weighted loss function to fight unbalanced dataset')

    args = parser.parse_args()

    provider = MicroscopyTrainDataLoader(args.csv_path, seed=args.seed)
    args.augmentation = (args.augmentation == 1)
    args.weight_sampler = (args.weight_sampler == 1)
    args.weight_loss = (args.weight_loss == 1)
    
    JSON_INPUT_PATH = "/workspace/src/experiments/microscopy/shallow-resnet50.json"
    

    with open(JSON_INPUT_PATH) as json_file:
        models = json.load(json_file)
    resnet50_4_2 = load_shallow_model('resnet50.layer4-2', models)
    resnet50_4_1 = load_shallow_model('resnet50.layer4-1', models)
    resnet50_4_0 = load_shallow_model('resnet50.layer4-0', models)
    resnet50_3_5 = load_shallow_model('resnet50.layer3-5', models)
    resnet50_3_4 = load_shallow_model('resnet50.layer3-4', models)
    resnet50_3_3 = load_shallow_model('resnet50.layer3-3', models)
    
    ensemble = MyEnsemble(resnet50_4_2, resnet50_4_1, resnet50_4_0, resnet50_3_5, resnet50_3_4, resnet50_3_3)
    for p in ensemble.parameters():
        p.requires_grad = False
    for p in ensemble.classifier.parameters():
        p.requires_grad = True
    
    run = ExperimentalRun(ensemble, provider, args, notes=args.notes, tags=['clef', 'microscopy', 'ensemble'])
    run.train()


if __name__ == '__main__':
    main()
