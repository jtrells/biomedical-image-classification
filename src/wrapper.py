import os
import pandas as pd
from dotenv import load_env
from pymongo import MongoClient, DESCENDING
from utils.FeatureExtractor import update_features
from utils.label_encoder import label_encoder_target
from utils.dimensionality_reduction import reduce_dimensions
from models.ResNetClass import ResNetClass
from pathlib import Path

def extract_and_update_features(model_path, csv_path, base_img_dir, label_col='label', batch_size=32, num_workers=16):
    model_path = Path(model_path)

    model = ResNetClass.load_from_checkpoint(model_path)
    df = pd.read_csv(csv_path, sep='\t')
    le_encoder, _ = label_encoder_target(df,target_col=label_col)

    try:
        update_features(model, le_encoder, csv_path, base_img_dir, label_col=label_col, seed=42, batch_size=batch_size, num_workers=num_workers)
        return True, None
    except Exception as e:
        return False, e


def get_data(taxonomy, classifier, reducer_name, version='latest', subset='all', num_dimensions=2):
    DB_CONNECTION = os.getenv('DB_CONN')

    client = MongoClient(DB_CONNECTION)
    db = client.classifiers

    if (version == 'latest'):
        rows = db.classifiers.find({'taxonomy': taxonomy,'classifier': classifier}).sort([('version', DESCENDING)])
        classifier_info = rows[0]
    else:
        classifier_info = db.classifiers.find_one({'taxonomy': taxonomy,'classifier': classifier, 'version': version})        
    
    csv_path = Path(classifier_info) / 'files' / taxonomy / classifier_info.dataset
    subset_col = 'split_set'
    df = pd.read_csv(csv_path, sep='\t')

    subset = None if subset == 'all' else subset
    features = reduce_dimensions(df, reducer_name, subset, subset_col, num_dimensions=num_dimensions)

    return features



