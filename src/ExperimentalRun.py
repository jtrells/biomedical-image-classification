import os
import torch
import wandb
import torch.nn as nn
import sys
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils import class_weight
from EarlyStopping import EarlyStopping
from tqdm import tqdm

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

class ExperimentalRun():
    def __init__(self, model, dataset_provider, config, notes="", tags=[]):
        self.config = config
        self._validate_config()
        
        self.run_name = self._start_wandb(notes, tags)
        self.run_path = self._create_output_dir(self.run_name)
        self._set_seeds()
        self._set_devices()
        self.model = self._set_model(model, self.device)
        wandb.watch(self.model)
        self.provider = dataset_provider
        
    def _start_wandb(self, notes, tags):
        wandb.init(
          project=self.config.project_name,
          notes=notes,
          tags=tags,
          reinit=True
        )
        wandb.run.save()
        wandb.config.update(self.config)
        
        return wandb.run.name
    
    def _create_output_dir(self, name):
        if name:
            output_run_path = os.path.join(self.config.out_dir, name)
            os.makedirs(output_run_path, exist_ok=False)
        else:
            raise Exception('run name cannot be null or empty')
        return output_run_path
    
    def _set_seeds(self):
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            
    def _set_devices(self):
        # use CUDA_VISIBLE_DEVICES=1 python myscript.py
        # for multi_gpu, the first GPU distributes the job on data parallelization
        # os['CUDA_VISIBLE_DEVICES'] does not seem to work on jupyter
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def _set_model(self, model, device):
        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        wandb.log({
            'Trainable Params': count_trainable_parameters(model),
            'Params': count_total_parameters(model),
        })
        return model.to(device)
    
    def train(self):
        train_dataset = self.provider.get_train_dataset(augmentation=self.config.augmentation)
        validation_dataset = self.provider.get_val_dataset()
        
        kwargs = {'num_workers': self.config.num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
        if not self.config.weight_sampler:
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, **kwargs)
        else:
            # TODO: forcing modality to be the label field. need to generalize this later
            train_labels = train_dataset.codec.transform(train_dataset.df['MODALITY'])
            _, train_class_counts = np.unique(train_labels, return_counts=True)
            # https://github.com/pytorch/tutorials/pull/236/files/37bfb4d2e063b538cb875043393cc596b195ebae
            weights = 1. / train_class_counts
            samples_weights = torch.from_numpy(weights[train_labels])
            sampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True)
            # sampler and shuffle are mutually exclusive
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler, **kwargs)
            
        valid_loader = DataLoader(validation_dataset, batch_size=self.config.test_batch_size, shuffle=False, **kwargs)
        
        early_stopping = EarlyStopping(patience=self.config.patience,
                                       verbose=True,
                                       path=os.path.join(self.run_path, 'checkpoint.pt'))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        if self.config.weight_loss:
            train_labels = train_dataset.codec.transform(train_dataset.df['MODALITY'])
            class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)
            class_weights = torch.Tensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_step(self.model, self.device, train_loader, optimizer, criterion, epoch, self.config.epochs)
            val_loss, val_acc, stop_execution = self.test_step(self.model, self.device, valid_loader, criterion, epoch, early_stopping)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if stop_execution:
                break

        best_val_epoch = early_stopping.get_best_val_epoch()
        wandb.log({
                "Best Val Loss": val_losses[best_val_epoch-1],
                "Best Val Acc": val_accs[best_val_epoch-1],
                "Best Train Loss ": train_losses[best_val_epoch-1],
                "Best Train Acc": train_accs[best_val_epoch-1],
                "Best Epoch": best_val_epoch,
            })
        
        saved_model_path = os.path.join(self.run_path, "weights")
        # avoid saving the weights per gpu module for more flexibility during loading
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), saved_model_path)
        else:
            torch.save(self.model.state_dict(), saved_model_path)
        wandb.join()

    def train_step(self, model, device, train_loader, optimizer, criterion, epoch, total_epochs):
        model.train()

        train_loss, n_iter = 0, 0
        total, correct = 0, 0

        with tqdm(total=len(train_loader), file=sys.stdout, leave=True, position=0) as tbar:
            tbar.set_description('training {epoch}/{total_epochs}'.format(epoch=epoch, total_epochs=total_epochs))
            for images, labels, _ in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total += labels.size(0)
                correct += torch.sum(predicted == labels.data)
                train_loss += loss.item()
                n_iter += 1
                
                tbar.update(1)
              
            epoch_acc = 100 * torch.true_divide(correct, total)
            average_loss = train_loss / n_iter
            
            tbar.set_postfix(loss=average_loss, accuracy=float(epoch_acc))

            wandb.log({
                "Train Loss": average_loss,
                "Train Acc": epoch_acc
            })
            return average_loss, epoch_acc        
            
    def test_step(self, model, device, test_loader, criterion, epoch, early_stopping):
        model.eval()

        valid_loss, valid_n_iter = 0, 0
        total, correct = 0, 0

        with tqdm(total=len(test_loader), file=sys.stdout, leave=True, position=0, desc='validation') as vbar:
            for images, labels, _ in test_loader:         
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                total += labels.size(0)
                correct += torch.sum(predicted == labels.data)
                valid_loss += loss.item()
                valid_n_iter += 1
                
                vbar.update(1)

            epoch_acc = 100 * torch.true_divide(correct, total)
            average_loss = valid_loss / valid_n_iter
            
            vbar.set_postfix(loss=average_loss, accuracy=float(epoch_acc))
            
            stop_execution = False
            early_stopping(average_loss, model, epoch)
            if early_stopping.early_stop:
                stop_execution = True

            wandb.log({
                "Test Accuracy": epoch_acc,
                "Test Loss": average_loss})

            return average_loss, epoch_acc, stop_execution
        

    def _validate_config(self):
        attrs = [
            'batch_size',         # Training batch size 
            'test_batch_size',    # Test/validation batch size
            'epochs',             # Number of max possible epochs in run; can be less
                                  # due to early stopping.
            'lr',                 # starting learning rate
            'seed',               # seed for results replicability
            'patience',           # epochs to wait before early stopping
            'num_workers',        # number of workers for data loading
            'num_output_classes', # output class for classification problem
            # 'dataset_path',       # path to input images
            # 'labels_path',        # path to CSV file with image labels (not using folder labels)
            'out_dir',            # where to save any output file
            'project_name',       # project name on wandb
            'architecture',       # network architecture
            'infra',              # infrastructure details
        ]
        
        missing = []
        for attr in attrs:
            if not hasattr(self.config, attr):
                missing.append(attr)                
        if len(missing) > 0:
            raise Exception('Configuration file missing attributes ' + ', '.join(missing))
        return