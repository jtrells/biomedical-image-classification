from pandas import concat as pd_concat
from pandas import read_csv
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def stratify(df, label_cols=['label', 'source'], n_splits=5):
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=False)
    for fold,(train_index, test_index) in enumerate(mskf.split(df, df[label_cols])):
        if fold == 0:   
            df.loc[test_index,'split_set'] = 'TEST'
            df.loc[train_index,'split_set'] = 'TRAIN'
    
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=False)
    df_train = df[df['split_set']=='TRAIN'].reset_index(drop = True)
    df_test  = df[df['split_set']=='TEST'].reset_index(drop = True)
    for fold,(train_index, test_index) in enumerate(mskf.split(df_train, df_train[['modality','source']])):
        if fold == 0:   
            df_train.loc[test_index,'split_set'] = 'VAL'
            df_train.loc[train_index,'split_set'] = 'TRAIN'
    df_concat = pd_concat.concat([df_train,df_test], axis=0).reset_index(drop=True)

    return df_concat


def convert_clef_dataset(csv_path, clef_mapping):
    df = read_csv(csv_path, sep='\t')
    df = df[df.source.isin(['clef13', 'clef16'])]
    # modality column has the original clef values
    prefix, mapping = clef_mapping

    df = df[df.modality[:len(prefix)] == prefix]
    df['modality'] = df['modality'].replace(mapping)    

    cols = ['img', 'source', 'img_path', 'caption', 'modality', 'split_set']
    df = df[cols]
    df = df.rename(columns={'modality': 'label'})    


def radiology_mapping():
    return 'DR', {
        'DRXR': 'X-RAY',
        'DRUS': 'ULTRASOUND',
        'DRAN': 'ANGIOGRAPHY',
        'DRMR': 'CT/MRI/PET',
        'DRPE': 'CT/MRI/PET',
        'DRCO': 'OTHER'
    }