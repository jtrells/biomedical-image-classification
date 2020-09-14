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
from microscopy import experiment

def get_experiments_per_layer(m, layer_name):
    if layer_name == 'layer4':
        count = len(m.layer4)
    elif layer_name == 'layer3':
        count = len(m.layer3)
    elif layer_name == 'layer2':
        count = len(m.layer2)
    elif layer_name == 'layer1':
        count = len(m.layer1)
    
    exps = []
    for i in range(count):
        exps.append(layer_name + '-'+ str(i))
    return exps



def main():
    
    models = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
    layers = ['layer4', 'layer3', 'layer2', 'layer1']
    
    for model_name in models:
        sample_model = experiment('whole', model_name, num_classes=4, pretrained=True)
        all_experiments = []
        for l in layers:
            all_experiments += get_experiments_per_layer(sample_model, l)
        all_experiments += ['whole']
    
        for exp in all_experiments:
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
            parser.add_argument('--augmentation', type=int, default='1', help='Use data augmentation')
            parser.add_argument('--weight-sampler', type=int, default='1', help='Weight sampling for unbalanced set')
            parser.add_argument('--weight-loss', type=int, default='1', help='Apply a weighted loss function to fight unbalanced dataset')

            args = parser.parse_args()

            # update args
            args.architecture = model_name
            args.experiment = exp
            args.lr = 5e-6
            args.epochs = 100
            args.patience = 100
            # all with weight loss and weight sampler and augmentation (see defaults)

            provider = MicroscopyTrainDataLoader(args.csv_path, seed=args.seed)
            model = experiment(args.experiment, args.architecture, num_classes=args.num_output_classes, pretrained= (args.pretrained == 1))
            args.augmentation = (args.augmentation == 1)
            args.weight_sampler = (args.weight_sampler == 1)
            args.weight_loss = (args.weight_loss == 1)

            run = ExperimentalRun(model, provider, args, notes=args.notes, tags=['clef', 'microscopy'])
            run.train()


if __name__ == '__main__':
    main()
