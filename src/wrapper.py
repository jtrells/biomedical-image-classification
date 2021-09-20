import pandas as pd
from numpy import vstack
from pymongo import DESCENDING
from utils.FeatureExtractor import update_features
from utils.dimensionality_reduction import reduce_dimensions, calc_neighborhood_hit, get_neighbors_by_index
from models.ResNetClass import ResNetClass
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def extract_and_update_features(model_path, parquet_path, base_img_dir, label_col='label', batch_size=32, num_workers=16):
    model = ResNetClass.load_from_checkpoint(model_path)

    update_features(model, parquet_path, base_img_dir, label_col=label_col, seed=42, batch_size=batch_size, num_workers=num_workers)
    # try:
    #     update_features(model, parquet_path, base_img_dir, label_col=label_col, seed=42, batch_size=batch_size, num_workers=num_workers)
    #     return True, None
    # except Exception as e:
    #     return False, e


def get_data(db, vil_path, taxonomy, classifier, reducer_name, version='latest', subset='all', num_dimensions=2, add_hits=True, label_col='label'):    
    # num_dimensions should be bigger than 1
    if (version == 'latest'):
        cursor = db.classifiers.find({'taxonomy': taxonomy,'classifier': classifier}).sort([('version', DESCENDING)])
        try:
            classifier_info = cursor.next()
        except: 
            raise Exception('no classifier available for parameters')
    else:
        classifier_info = db.classifiers.find_one({'taxonomy': taxonomy,'classifier': classifier, 'version': version})        
    
    parquet_path = Path(vil_path) / 'files' / taxonomy / classifier_info['dataset']
    subset_col = 'split_set'
    df = pd.read_parquet(parquet_path)

    # if aiming for a subset, reduce the dataframe
    subset = None if subset == 'all' else subset
    if subset != None:
        df = df[df[subset_col]==subset].reset_index(drop = True)
    features = vstack(df.features.values)

    embeddings = reduce_dimensions(features, reducer_name, num_dimensions=num_dimensions)
    le = LabelEncoder().fit(df[label_col].unique())
    # get the neighborhood hit in feature space
    n_hits = calc_neighborhood_hit(df, features, le, n_neighbors=6, label_col=label_col) if add_hits else None
    
    df['x'] = embeddings[:, 0]
    df['y'] = embeddings[:, 1]
    df['hits'] = vstack(n_hits)

    if num_dimensions > 2:
        print("dimensions > 2 but only retrieving the first three dimensions")
        df['z'] = embeddings[:, 2]
    
    return df


def get_figure_neighbors(db, vil_path, taxonomy, classifier, version, img_path, n_neighbors):
    if (version == 'latest'):
        cursor = db.classifiers.find({'taxonomy': taxonomy,'classifier': classifier}).sort([('version', DESCENDING)])
        try:
            classifier_info = cursor.next()
        except: 
            raise Exception('no classifier available for parameters')
    else:
        classifier_info = db.classifiers.find_one({'taxonomy': taxonomy,'classifier': classifier, 'version': version})            
    parquet_path = Path(vil_path) / 'files' / taxonomy / classifier_info['dataset']

    df = pd.read_parquet(parquet_path)    
    index = df[df.img_path==img_path].index[0]

    return get_neighbors_by_index(df, index, n_neighbors=n_neighbors)
