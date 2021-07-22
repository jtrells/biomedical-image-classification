from pandas import concat as pd_concat
from pandas import read_csv
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# There are no modalities for 

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


def convert_clef_openi_dataset(csv_path, dataset, mapping):
    df = read_csv(csv_path, sep='\t')
    if dataset == 'clef':
        df = df[df.source.isin(['clef13', 'clef16'])]
    else:
        df = df[df.source.isin(['openi'])]
    # modality column has the original clef values
    df = df[df.modality.isin(list(mapping.keys()))]
    df['modality'] = df['modality'].replace(mapping)

    cols = ['img', 'source', 'img_path', 'caption', 'modality', 'split_set']
    df = df[cols]
    df = df.rename(columns={'modality': 'label'})
    return df


def get_radiology_dataset(csv_path):
    mappings = radiology_mapping()
    df_clef  = convert_clef_openi_dataset(csv_path, 'clef', mappings['clef'])
    df_openi = convert_clef_openi_dataset(csv_path, 'openi', mappings['openi'])
    df = pd_concat[df_clef, df_openi]
    return stratify(df)


def radiology_mapping():
    return {
        'clef': {
            'DRXR': 'X-RAY',
            'DRUS': 'ULTRASOUND',
            'DRAN': 'ANGIOGRAPHY',
            'DRMR': 'CT/MRI/PET',
            'DRPE': 'CT/MRI/PET',
            'DRCO': 'OTHER'
        }, 
        'openi': {
            'DRXR': 'X-RAY',
            'DRCT': 'CT/MRI/PET',
            'DRUS': 'ULTRASOUND',
            'DRMR': 'CT/MRI/PET'
        }
    }