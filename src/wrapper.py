import pandas as pd
from utils.FeatureExtractor import update_features
from utils.label_encoder import label_encoder_target
from models.ResNetClass import ResNetClass
from pathlib import Path

def extract_and_update_features(model_path, csv_path, base_img_dir, label_col='label', batch_size=32, num_workers=16):
    model_path = Path(model_path)

    model = ResNetClass.load_from_checkpoint(model_path)
    df = pd.read_csv(csv_path, sep='\t')
    le_encoder, _ = label_encoder_target(df,target_col=label_col)

    updated_df = update_features(model, le_encoder, csv_path, base_img_dir, seed=42, batch_size=batch_size, num_workers=num_workers)