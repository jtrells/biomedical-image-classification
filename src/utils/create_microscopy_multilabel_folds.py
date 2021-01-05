'''
Take the input captions file in multilabel format: 'caption', 'modality1', 'modality2', 'modality3'
and divide the train dataset in k-folds. As the y value is multi-label, we use two approaches:
1. KFOLD: Join the multi-labels in one new column and stratify as a multi-class problem (sklearn)
2. KFOLD_MULTI: Use iterative multi-label stratification (scikit-multilearn)

Reproducibility:
Set random_state=443 and 5 folds. However, IterativeStratification (KFOLD_MULTI)
interface does not use random_state as its base class does not shuffle values. Therefore, please
refer to the /data folder for the originally created file. KFOLD column is indeed reproducible.

Warning: 
Some unique multi-label combinations have fewer than 5 samples, thus, you may see a warning from
sklearn model selection.

# !pip install scikit-multilearn
# !pip install arff
'''

import pandas as pd
from sklearn import model_selection
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
from sklearn.preprocessing import LabelEncoder

random_state = 443
n_splits = 5
output_dir = '/workspace/data'
csv_path = '/workspace/data/multilabel-captions.csv'
multilabel_cols = ['DMEL', 'DMFL', 'DMLI', 'DMTR']

def labels_to_str(y):
    # Combine multi-labels to unique y column
    y_new = LabelEncoder().fit_transform([str(l) for l in y])
    return y_new

if __name__ == "__main__":
    df = pd.read_csv(csv_path, sep='\t')
    df.loc[:, "KFOLD"] = -1               # transforming multilabel to string as a new unique label, so y is [N,]
    df.loc[:, "KFOLD_MULTI"] = -1         # feeding multilabel y ([N, 4]) to a iterative stratification, random_state not enabled.
    
    df_train = df[df.SET == 'TRAIN']
    df_test  = df[df.SET == 'TEST']
    
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True) # randomize dataset
        
    skf = model_selection.StratifiedKFold(n_splits=n_splits)
    skf_multilabel = IterativeStratification(n_splits=n_splits, order=1) # random_state=random_state
    
    x = df_train.CAPTION.values
    y = df_train[multilabel_cols].values
    y_unique = labels_to_str(y)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=x, y=y_unique)):
        df_train.loc[val_idx, "KFOLD"] = fold
    
    for fold, (train_idx, val_idx) in enumerate(skf_multilabel.split(x, y)):
        df_train.loc[val_idx, "KFOLD_MULTI"] = fold    
    
    df_new = pd.concat([df_train, df_test]).reset_index(drop=True)
    df_new[['CAPTION', 'SET', 'KFOLD', 'KFOLD_MULTI'] + multilabel_cols]
    
    df_new.to_csv(f"{output_dir}/microscopy_captions_multilabel_kfolds.csv", index=False, sep='\t')
    
    