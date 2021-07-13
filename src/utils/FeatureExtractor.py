import torch
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
import os
import sys
module_path = "../dataset"
#os.environ["WANDB_SILENT"] = "true" # Environment Variable to make wandb silent
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset.ImageDataModule import ImageDataModule
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from dataset.ImageDataset import ImageDataset, EvalImageDataset
from torch.utils.data import DataLoader
from pathlib import Path


def get_vector_representation(data_loader, model, device):
    model.to(device)
    # Put the model in eval mode
    model.eval()
    # List for store final predictions
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            if isinstance(data, tuple):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                predictions = model(data[0])
            else:
                predictions = model(data.to(device))
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return np.vstack((final_predictions))[:,:,0,0]


def calc_neighborhood_hit(df,x_col,y_col, n_neighbors=6,column_label='target'):    
    projections = [[i, j] for (i, j) in zip(df[x_col], df[y_col])]
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(projections)
    le = preprocessing.LabelEncoder().fit(df[column_label].unique())
    n_hits = []
    for neighborhood in neigh.kneighbors(projections, n_neighbors + 1, return_distance=False):
        labels  = le.transform(df.iloc[neighborhood][column_label].values)
        targets = [labels[0]] * (len(labels) - 1) 
        n_hit = np.mean(targets == labels[1:])
        n_hits.append(n_hit)
    return n_hits


# +
def prepare_projection(model ,le_encoder,DATA_PATH,BASE_IMG_DIR,SEED,CLASSF ='higher_modality' ,VERSION = 1):
    ## Get Feature Vector for the dataset
    df        = pd.read_csv(DATA_PATH,sep = '\t')    
    transform = [transforms.ToPILImage(),
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(model.hparams['mean_dataset'], model.hparams['std_dataset'])
                ]
    transform = transforms.Compose(transform)
    # 1. Train Dataloader
    dm = ImageDataModule    ( batch_size  = 32,
                              label_encoder    = le_encoder,
                              data_path        = str(DATA_PATH), 
                              base_img_dir     = str(BASE_IMG_DIR),
                              seed             = SEED,   
                              image_transforms = [transform,transform,transform],
                              num_workers      = 72,
                              target_class_col ='split_set',
                              modality_col     ='target',
                              path_col         ='img_path',
                              shuffle_train    = False) # Not Shuffling Train
    dm.prepare_data()
    dm.setup()

    # 3. Model
    df_train,df_val,df_test = df[df['split_set']=='TRAIN'].reset_index(drop = True),df[df['split_set']=='VAL'].reset_index(drop = True),df[df['split_set']=='TEST'].reset_index(drop = True)
    fe_model = model.feature_extraction()
    train_dataloader,val_dataloader,test_dataloader = dm.train_dataloader(),dm.val_dataloader(),dm.test_dataloader()
    # Getting the feature matrix
    print('Feature Vector for training: ')
    fe_matrix_train        = get_vector_representation(train_dataloader,fe_model,'cuda')
    print('Feature Vector for Validation: ')
    fe_matrix_val          = get_vector_representation(val_dataloader,fe_model,'cuda')
    print('Feature Vector for Test: ')
    fe_matrix_test          = get_vector_representation(test_dataloader,fe_model,'cuda')
    
    df_train['feature_vector'] = list(fe_matrix_train)
    df_val['feature_vector']   = list(fe_matrix_val)
    df_test['feature_vector']  = list(fe_matrix_test)
    
    # Doing Dimensional reduction
    
    print('***** PCA *****')
    pca = PCA(n_components=2,random_state = SEED)
    pca.fit(fe_matrix_train)
    embedding_train  = pca.transform(fe_matrix_train)
    embedding_val   = pca.transform(fe_matrix_val)
    embedding_test  = pca.transform(fe_matrix_test)
    
    df_train['pca_x'], df_train['pca_y'] = embedding_train[:,0],embedding_train[:,1]
    df_val['pca_x']  , df_val['pca_y']   = embedding_val[:,0]  ,embedding_val[:,1]
    df_test['pca_x'] , df_test['pca_y']  = embedding_test[:,0]  ,embedding_test[:,1]
    
    # Adding the PCA n_hits
    df_train['pca_hits'] = calc_neighborhood_hit(df_train,'pca_x','pca_y', n_neighbors=6)
    df_val['pca_hits'] = calc_neighborhood_hit(df_val,'pca_x','pca_y', n_neighbors=6)
    df_test['pca_hits'] = calc_neighborhood_hit(df_test,'pca_x','pca_y', n_neighbors=6)
    
    
    del embedding_train,embedding_val,embedding_test
    
    
    print('***** UMAP *****')
    reducer = umap.UMAP(random_state=SEED)
    reducer.fit(fe_matrix_train)
    embedding_train  = reducer.transform(fe_matrix_train)
    embedding_val    = reducer.transform(fe_matrix_val)
    embedding_test   = reducer.transform(fe_matrix_test)
    
    df_train['umap_x'], df_train['umap_y'] = embedding_train[:,0],embedding_train[:,1]
    df_val['umap_x']  , df_val['umap_y']   = embedding_val[:,0]  ,embedding_val[:,1]
    df_test['umap_x'] , df_test['umap_y']  = embedding_test[:,0]  ,embedding_test[:,1]
    
    
    # Adding the UMAP n_hits
    df_train['umap_hits'] = calc_neighborhood_hit(df_train,'umap_x','umap_y', n_neighbors=6)
    df_val['umap_hits'] = calc_neighborhood_hit(df_val,'umap_x','umap_y', n_neighbors=6)
    df_test['umap_hits'] = calc_neighborhood_hit(df_test,'umap_x','umap_y', n_neighbors=6)
    
    del embedding_train,embedding_val,embedding_test
    
    df_total = pd.concat([df_train,df_val,df_test],axis = 0).reset_index(drop = True)
    #df_total.to_csv(f'/mnt/artifacts/projections/higher_modality_v{VERSION}.csv',sep = '\t',index =False)
    df_total.to_parquet(f'/mnt/artifacts/projections/{CLASSF}_v{VERSION}.parquet',index =False)
    #with open(f'/mnt/artifacts/projections/higher_modality_features_v{VERSION}.pkl','wb') as f: pickle.dump(np.concatenate((fe_matrix_train, fe_matrix_val,fe_matrix_test), axis=0), f)
    del fe_matrix_train, fe_matrix_val,fe_matrix_test
    
def evaluate_projection(DF_UL,model,prev_train_dataset,SEED=42):
    df = pd.read_parquet(prev_train_dataset)
    fe_matrix_train = np.vstack(df[df['split_set']=='TRAIN'].reset_index(drop = True)['feature_vector'].values)
    
    fe_model = model.feature_extraction()
    val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(model.hparams['mean_dataset'].numpy(),model.hparams['std_dataset'].numpy())
                    ])

    DF_UL_dataset    = EvalImageDataset (DF_UL,image_transform=val_transform,path_col='img_path',base_img_dir = Path('/mnt/'))
    DF_UL_dataloader = DataLoader(dataset     = DF_UL_dataset,
                                  batch_size  = 32,
                                  shuffle     = False,
                                  num_workers = 72)
    
    fe_matrix_unlabeled = get_vector_representation(DF_UL_dataloader,fe_model,'cuda')
    # PCA
    print('Generating PCA coordinates...')
    pca = PCA(n_components=2,random_state = SEED)
    pca.fit(fe_matrix_train)
    embedding_unlabeled = pca.transform(fe_matrix_unlabeled)
    DF_UL['pca_x'] = embedding_unlabeled[:,0]
    DF_UL['pca_y'] = embedding_unlabeled[:,1]
    DF_UL['pca_hits'] = calc_neighborhood_hit(DF_UL,'pca_x','pca_y', n_neighbors=6,column_label = 'target_predicted')
    
    #UMAP
    print('Generating Umap coordinates...')
    reducer = umap.UMAP(random_state=SEED)
    reducer.fit(fe_matrix_train)
    embedding_unlabeled = reducer.transform(fe_matrix_unlabeled)
    DF_UL['umap_x'] = embedding_unlabeled[:,0]
    DF_UL['umap_y'] = embedding_unlabeled[:,1]
    DF_UL['umap_hits'] = calc_neighborhood_hit(DF_UL,'umap_x','umap_y', n_neighbors=6,column_label = 'target_predicted')

    del fe_matrix_train,fe_matrix_unlabeled,embedding_unlabeled
