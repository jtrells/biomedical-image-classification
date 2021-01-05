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
from dataset.MultilabelDataModule import MultilabelDataModule    
from utils.caption_utils import load_glove_matrix, load_bioword_matrix                      

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger    

from dataset.utils import clean_str

def main():
    parser = argparse.ArgumentParser(description='Microscopy Classification CLEF - only captions')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')    
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=443, metavar='S', help='random seed (default: 443)')
    parser.add_argument('--patience', type=int, default=100, metavar='N', help='early stopping patience')
    parser.add_argument('--num-workers', type=int, default=72, metavar='N', help='processors for data loading tasks')
    parser.add_argument('--vocab-size', type=int, default=20000, metavar='N')
    parser.add_argument('--max-words-sentence', type=int, default=300, metavar='N')
    parser.add_argument('--word-dimension', type=int, default=200, metavar='N') # 200 for bioword
    parser.add_argument('--num-classes', type=int, default=4, metavar='N')
    parser.add_argument('--num-filters', type=int, default=100, metavar='N')
    parser.add_argument('--base-path', type=str, default='/workspace/data', metavar='N')
    parser.add_argument('--output-dir', type=str, default='./outputs', metavar='N')
    parser.add_argument('--project', type=str, default='biomedical-multimodal', metavar='N')
    parser.add_argument('--kfolds', type=int, default=5, metavar='N')
    parser.add_argument('--tags', nargs='+')
        
    args = parser.parse_args()
    
    BASE_PATH = Path(args.base_path)
    DATA_PATH = BASE_PATH / 'microscopy_captions_multilabel_kfolds.csv' #'multilabel-captions.csv' #multimodality_classification.csv'
    #EMBEDDINGS = BASE_PATH / 'embeddings'
    EMBEDDINGS = BASE_PATH / 'biosentvec' / 'BioWordVec_PubMed_MIMICIII_d200.vec.bin'
    OUTPUT_DIR = Path(args.output_dir)
    BASE_IMG_DIR = BASE_PATH
    
    filters = [75]
    lrs = [3e-4]
    max_words_sentence = [150]
    tags = ['script', 'captions', 'bioword', 'cleaned', 'multi-label', 'kfold']   
    
    # hardcode
    args.kfolds = 1
    
    for fltr in filters:
        for lr in lrs:
            for mwps in max_words_sentence:
                dm = MultilabelDataModule(args.batch_size,
                                            str(DATA_PATH),
                                            args.vocab_size,
                                            mwps,
                                            num_workers=args.num_workers,
                                            random_state=args.seed,
                                            kfold_col='KFOLD',
                                            preprocess_fn=clean_str)
                dm.prepare_data() 

                for k_fold_idx in np.arange(args.kfolds):
                    wandb_logger = WandbLogger(project=args.project, tags=tags)
                    wandb_logger.experiment.save()
                    print(wandb_logger.experiment.name)
                    output_run_path = OUTPUT_DIR / wandb_logger.experiment.name 
                    os.makedirs(output_run_path, exist_ok=False)
                    
                    print("K: {0}".format(k_fold_idx))                    
                    dm.setup(k_fold_idx=k_fold_idx)
                    print("Current vocabulary size: {0}".format(dm.vocab_size))
    
                    #embeddings_matrix = load_glove_matrix(EMBEDDINGS, args.word_dimension, dm.vocab_size, dm.word_index)  
                    embeddings_matrix = load_bioword_matrix(EMBEDDINGS, dm.vocab_size, dm.word_index)
                    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=args.patience, verbose=True, mode='min')
                           
                    model = CaptionModalityClassifier(
                         max_input_length=mwps,
                         vocab_size=dm.vocab_size,
                         embedding_dim=args.word_dimension,
                         filters=fltr,
                         embeddings=embeddings_matrix,
                         num_classes=args.num_classes,
                         train_embeddings=True,
                         is_multilabel=True,
                         lr=lr)

                    trainer = Trainer(gpus=1,
                          max_epochs=args.epochs,
                          default_root_dir=str(output_run_path),
                          early_stop_callback=early_stop_callback,
                          logger=wandb_logger)
                    trainer.fit(model, dm)
                    trainer.save_checkpoint(str(output_run_path / 'final.pt'))
                print("end-cross validation #####################################################################")

if __name__ == '__main__':
    main()
