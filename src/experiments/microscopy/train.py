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
    if args.architecture == "efficient-net":
        model = get_model("efficient-net", "efficient-net", num_classes=args.num_output_classes, pretrained= (args.pretrained == 1))
    else:
        model = experiment(args.experiment, args.architecture, num_classes=args.num_output_classes, pretrained= (args.pretrained == 1))
    args.augmentation = (args.augmentation == 1)
    args.weight_sampler = (args.weight_sampler == 1)
    args.weight_loss = (args.weight_loss == 1)
    
    run = ExperimentalRun(model, provider, args, notes=args.notes, tags=['clef', 'microscopy'])
    run.train()


if __name__ == '__main__':
    main()
