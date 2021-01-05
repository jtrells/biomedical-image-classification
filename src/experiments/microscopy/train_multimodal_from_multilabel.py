import os
import sys
print(os.getcwd())
module_path = os.path.abspath(os.path.join('../../../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from pathlib import Path
import argparse

from models.CaptionModalityClassifier import CaptionModalityClassifier
from models.MultiModalityClassifier import MultiModalityClassifier
from dataset.MultimodalityDataModule import MultimodalityDataModule
from experiments.microscopy.microscopy import experiment, get_model

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger    

import json
import torch

def main():
    parser = argparse.ArgumentParser(description='Microscopy Classification CLEF - multimodal')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')    
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')    
    parser.add_argument('--seed', type=int, default=443, metavar='S', help='random seed (default: 443)')
    parser.add_argument('--patience', type=int, default=5, metavar='N', help='early stopping patience')
    parser.add_argument('--num-workers', type=int, default=72, metavar='N', help='processors for data loading tasks')    
    parser.add_argument('--num-classes', type=int, default=4, metavar='N')
    parser.add_argument('--num-filters', type=int, default=128, metavar='N')
    parser.add_argument('--base-path', type=str, default='/workspace/data', metavar='N')
    parser.add_argument('--vocab-size', type=int, default=20000, metavar='N')
    parser.add_argument('--output-dir', type=str, default='./outputs', metavar='N')
    parser.add_argument('--project', type=str, default='biomedical-multimodal', metavar='N')
    parser.add_argument('--tags', nargs='+')
    parser.add_argument('--vision-outputs-path', type=str, default='/workspace/outputs', metavar='N')
    parser.add_argument('--vision-dict-path', type=str, default='/workspace/src/experiments/microscopy/shallow-resnet152.json', metavar='N')
    parser.add_argument('--vision-model', type=str, default='resnet152.layer3-11')
    parser.add_argument('--text-model-path', type=str, default='/workspace/nb/outputs/bumbling-dragon-180/final.pt')
    parser.add_argument('--max-words-sentence', type=int, default=500, metavar='N')
    parser.add_argument('--num-vision-outputs', type=int, default=1024, metavar='N')    
    
    args = parser.parse_args()
    
    BASE_PATH = Path(args.base_path)
    DATA_PATH = BASE_PATH / 'multimodality_classification.csv'
    OUTPUT_DIR = Path(args.output_dir)
    BASE_IMG_DIR = BASE_PATH
    TEXT_MODEL_PATH = args.text_model_path
    MAX_NUMBER_WORDS=20000
    
    lrs = [5e-5]    
    tags = ['script', 'glove', 'multimodal', 'multi-label']
    
    with open(args.vision_dict_path) as json_file:
        vision_models = json.load(json_file)
    vision_model_name, vision_experiment_name = args.vision_model.split('.')
    vision_model = get_model(vision_model_name, "shallow", 4, layers=vision_models[args.vision_model]['layers'], pretrained=True)
    
    checkpoint = torch.load(args.vision_outputs_path + '/{0}/checkpoint.pt'.format(vision_models[args.vision_model]['id']))
    vision_model.load_state_dict(checkpoint)
    
    from utils.caption_utils import load_glove_matrix
    EMBEDDINGS = BASE_PATH / 'embeddings'
    WORD_DIMENSION = 300 

    dm = MultimodalityDataModule(args.batch_size, str(DATA_PATH), args.vocab_size, args.max_words_sentence,
                                 str(BASE_IMG_DIR), num_workers=args.num_workers)
    dm.prepare_data()
    dm.setup()
    
    if dm.vocab_size < MAX_NUMBER_WORDS:
        MAX_NUMBER_WORDS = dm.vocab_size

#     embeddings_matrix = load_glove_matrix(EMBEDDINGS, WORD_DIMENSION, MAX_NUMBER_WORDS, dm.word_index)     
    
    for lr in lrs:            

        print("Current vocabulary size: {0}".format(dm.vocab_size))
        train_dataloader = dm.train_dataloader()
        train_dataset    = train_dataloader.dataset
        target_classes   = train_dataset.le.classes_
        print("Classes: {0}".format(target_classes))                

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=args.patience, verbose=True, mode='min')

        wandb_logger = WandbLogger(project=args.project, tags=tags)
        wandb_logger.experiment.save()
        print(wandb_logger.experiment.name)
        output_run_path = OUTPUT_DIR / wandb_logger.experiment.name 
        os.makedirs(output_run_path, exist_ok=False)

        text_model = CaptionModalityClassifier.load_from_checkpoint(filters=args.num_filters,
                                                            embedding_dim=300,
                                                            vocab_size=8938,#dm.vocab_size,
                                                            num_classes=4,
                                                            train_embedding=False,
                                                            max_input_length=args.max_words_sentence,
                                                            is_multilabel=True,
                                                            target_classes=None,
                                                            checkpoint_path=TEXT_MODEL_PATH)
        model = MultiModalityClassifier(text_model, vision_model,
                                        num_filters=args.num_filters, num_vision_outputs=args.num_vision_outputs)        
        
        trainer = Trainer(gpus=1,
              max_epochs=args.epochs,
              default_root_dir=str(output_run_path),
              early_stop_callback=early_stop_callback,
              logger=wandb_logger)
        trainer.fit(model, dm)
        trainer.save_checkpoint(str(output_run_path / 'final.pt'))


if __name__ == '__main__':
    main()
