# 1. General Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from argparse import ArgumentParser
from os import makedirs, listdir
# 2. DataModule & Class Libraries
from utils.label_encoder import label_encoder_target
from utils.calc_stat import calc_dataset_mean_std
from dataset.ImageDataModule import ImageDataModule
from dataset.ImageDataset import ImageDataset
from models.EfficientNetClass import EfficientNetClass
from models.ResNetClass import ResNetClass
# 3. Pytorch & Pytorch Lightning Libraries
from pytorch_lightning import Trainer,seed_everything
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
# 4. Wandb Tracker Experiements
import wandb

class Run():
    def __init__(self):
        args = self._read_arguments()
        self.data_path = Path(args.dataset_filepath)
        self.base_img_dir = Path(args.images_path)
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.project = args.project
        self.lr = args.lr
        self.gpus = args.gpus
        self.classifier = args.classifier_name
        self.metric_monitor = 'val_avg_loss'
        self.mode = 'min'
        self.extension = '.pt'
        self.batch_size = args.batch_size
        self.label_col = 'higher_modality'
        self.split_col = 'split_set'
        self.img_path_col = 'img_path'        

        self.output_dir= Path(args.output_dir) / self.classifier
        makedirs(self.output_dir, exist_ok=True)
        self.version = self._get_version()        

        df = pd.read_csv(self.data_path, sep='\t')
        self.le, dict_label = label_encoder_target(df)

        pass

    def _read_arguments(self):
        parser = ArgumentParser(description="biomedical image classifier")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=0, help='if 0, use patience')
        parser.add_argument('--seed', type=int, default=443)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--dataset_filepath', type=str, help='location of .csv file')
        parser.add_argument('--images_path', type=str, help='root folder for training images')
        parser.add_argument('--output_dir', type=str, help='where to save results')
        parser.add_argument('--classifier_name', type=str)
        parser.add_argument('--project', type=str)
        parser.add_argument('--gpus', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--model_name', type=str, default='resnet101')

        return parser.parse_args()

    def _get_version(self):
        models = [x for x in listdir(self.output_dir) if x[:-3] == '.pt']
        return len(models) + 1        

    def _calculate_dataset_stats(self):
        transform_list = [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
        ]
        transform  = transforms.Compose(transform_list)
        train_dataset   = ImageDataset( self.data_path,
                                        self.le,
                                        str(self.base_img_dir),
                                        'TRAIN',
                                        image_transform=transform,
                                        label_name=self.label_col,
                                        target_class_col=self.split_col,
                                        path_col=self.img_path_col)

        mean, std = calc_dataset_mean_std(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return mean, std

    def _get_train_val_transformations(self, mean, std):
        train_transform = [
                            transforms.ToPILImage(),
                            transforms.Resize((256, 256)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation(15),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean,std)
                        ]
        train_transform  = transforms.Compose(train_transform )

        val_transform = [
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
                        ]
        val_transform = transforms.Compose(val_transform)

        test_transform = [
                  transforms.ToPILImage(),
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean,std)
                  ]
        test_transform = transforms.Compose(test_transform)

        return train_transform, val_transform, test_transform

    def run(self):
        seed_everything(self.seed)
        mean, std = self._calculate_dataset_stats()
        train_transform, val_transform, test_transform = self._get_train_val_transformations(mean, std)

        wandb.init()
        wandb_logger = WandbLogger(project=self.project, reinit=True)
        wandb_logger.experiment.save()

        output_run_path = self.output_dir
        # makedirs(output_run_path, exist_ok=False)

        # setup data
        dm = ImageDataModule( batch_size     = self.batch_size,
                            label_encoder    = self.le,
                            data_path        = str(self.data_path), 
                            base_img_dir     = str(self.base_img_dir),
                            seed             = self.seed,   
                            image_transforms = [train_transform,val_transform,test_transform],
                            num_workers      = self.num_workers,
                            target_class_col = self.split_col, # TRAIN, VAL, TEST
                            modality_col     = self.label_col,
                            path_col         = self.img_path_col)
        dm.prepare_data()
        dm.setup()
        dm.set_seed()        

        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop_callback = EarlyStopping(
            monitor   = self.metric_monitor,
            min_delta = 0.0,
            patience  = 5,
            verbose   = True,
            mode      = self.mode
        )
        
        checkpoint_callback = ModelCheckpoint(dirpath    = output_run_path,
                                            filename   = f'{self.classifier}_{self.version}',
                                            monitor    = self.metric_monitor,
                                            mode       = self.mode,
                                            save_top_k = 1)
        checkpoint_callback.FILE_EXTENSION = self.extension

        num_classes = len(self.le.classes_)
        
        model = ResNetClass    (name            = wandb.config.name,
                                num_classes     = num_classes,
                                pretrained      = True,
                                fine_tuned_from = 'whole',
                                lr              = self.lr,
                                metric_monitor  = self.metric_monitor,
                                mode_scheduler  = self.mode,
                                class_weights   = dm.class_weights,
                                mean_dataset    = mean,
                                std_dataset     = std)
        if self.version > 1:
            # model = ResNetClass.load_from_checkpoint(self.output_dir/f'{self.classifier}_{self.version}.{self.extension}')
            # model.class_weights = dm.class_weights
            # model.mean_dataset = mean
            # model.std_dataset = std
            # self.save_hyperparameters("class_weights","mean_dataset","std_dataset")            
            checkpoint = torch.load(self.output_dir/f'{self.classifier}_{self.version-1}.{self.extension}')
            model.load_state_dict(checkpoint['state_dict'])

        max_epochs = 100 if self.epochs == 0 else self.epochs
        callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]
        trainer = Trainer(gpus=self.gpus,
                    max_epochs=max_epochs,                  
                    callbacks=callbacks,
                    deterministic= True,
                    logger=wandb_logger,
                    num_sanity_val_steps=0)
        trainer.fit(model, dm)
        
        wandb.finish()
        return f'{self.classifier}_{self.version}.{self.extension}'


if __name__ == '__main__':
    run = Run()
    run.run()