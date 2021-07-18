import os
import pandas as pd
from pymongo import DESCENDING
from utils.FeatureExtractor import update_features
from utils.dimensionality_reduction import reduce_dimensions
from models.ResNetClass import ResNetClass
from pathlib import Path

def extract_and_update_features(model_path, parquet_path, base_img_dir, label_col='label', batch_size=32, num_workers=16):
    model = ResNetClass.load_from_checkpoint(model_path)

    update_features(model, parquet_path, base_img_dir, label_col=label_col, seed=42, batch_size=batch_size, num_workers=num_workers)
    # try:
    #     update_features(model, parquet_path, base_img_dir, label_col=label_col, seed=42, batch_size=batch_size, num_workers=num_workers)
    #     return True, None
    # except Exception as e:
    #     return False, e


def get_data(db, vil_path, taxonomy, classifier, reducer_name, version='latest', subset='all', num_dimensions=2):    
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

    subset = None if subset == 'all' else subset
    features = reduce_dimensions(df, reducer_name, subset, subset_col, num_dimensions=num_dimensions)

    return features



