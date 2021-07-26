from pandas import concat as pd_concat
from pandas import read_csv
from os import path, read
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# There are no modalities for


def stratify(df, label_cols=['label', 'source'], n_splits=5):
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=False)
    for fold, (train_index, test_index) in enumerate(mskf.split(df, df[label_cols])):
        if fold == 0:
            df.loc[test_index, 'split_set'] = 'TEST'
            df.loc[train_index, 'split_set'] = 'TRAIN'

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=False)
    df_train = df[df['split_set'] == 'TRAIN'].reset_index(drop=True)
    df_test = df[df['split_set'] == 'TEST'].reset_index(drop=True)
    for fold, (train_index, test_index) in enumerate(mskf.split(df_train, df_train[label_cols])):
        if fold == 0:
            df_train.loc[test_index, 'split_set'] = 'VAL'
            df_train.loc[train_index, 'split_set'] = 'TRAIN'
    df_concat = pd_concat([df_train, df_test], axis=0).reset_index(drop=True)

    return df_concat


def convert_clef_openi_dataset(csv_path, dataset, mapping):
    df = read_csv(csv_path, sep='\t')
    if dataset == 'clef':
        df = df[df.source.isin(['clef13', 'clef16'])]
    elif dataset == 'openi':
        df = df[df.source.isin(['openi'])]
    # modality column has the original clef values
    df = df[df.modality.isin(list(mapping.keys()))]
    df['modality'] = df['modality'].replace(mapping)

    cols = ['img', 'source', 'img_path', 'caption', 'modality', 'split_set']
    df = df[cols]
    df = df.rename(columns={'modality': 'label'}).reset_index(drop=True)
    return df


def convert_tinman_dataset(csv_path, mapping):
    df = read_csv(csv_path, sep='\t')
    df = df[df.modality.isin(list(mapping.keys()))]
    df['modality'] = df['modality'].replace(mapping)
    cols = ['img', 'source', 'img_path', 'caption', 'modality', 'split_set']
    df = df[cols]
    df = df.rename(columns={'modality': 'label'}).reset_index(drop=True)
    return df


def get_radiology_dataset(csv_path):
    mappings = radiology_mapping()
    df_clef = convert_clef_openi_dataset(csv_path, 'clef', mappings['clef'])
    df_openi = convert_clef_openi_dataset(csv_path, 'openi', mappings['openi'])
    df = pd_concat([df_clef, df_openi], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_experimental_dataset(clef_csv_path, gel_csv_path, gel_base_path, plate_csv_path, plate_base_path, tinman_path):
    mapping = experimental_mapping()
    df_clef = convert_clef_openi_dataset(
        clef_csv_path, 'clef', mapping['clef'])

    df_gel = read_csv(gel_csv_path)
    df_gel['img'] = df_gel["filepath"].str.split("/", expand=True)[2]
    df_gel['img_path'] = gel_base_path + df_gel['filepath']
    df_gel['label'] = 'GEL'
    df_gel['source'] = 'PUBMED'
    df_gel['caption'] = ''

    df_plates = read_csv(plate_csv_path)
    df_plates['img'] = df_plates["filepath"].str.split("/", expand=True)[2]
    df_plates['img_path'] = plate_base_path + df_plates['filepath']
    df_plates['label'] = 'PLATE'
    df_plates['source'] = 'PUBMED'
    df_plates['caption'] = ''

    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])

    columns = ['img', 'label', 'source', 'img_path', 'caption']
    df = pd_concat([df_clef[columns], df_gel[columns],
                   df_plates[columns], df_tinman[columns]], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_gel_dataset(tinman_path):
    # Taxonomy comes from the labeling interface
    mapping = gel_mapping()
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])
    df = stratify(df_tinman)
    return df


def get_electron_dataset(clef_path, tinman_path):
    mapping = electron_mapping()
    df_clef = convert_clef_openi_dataset(clef_path, 'clef', mapping['clef'])
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])
    df = pd_concat([df_clef, df_tinman], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_microscopy_dataset(clef_path, tinman_path):
    mapping = microscopy_mapping()
    df_clef = convert_clef_openi_dataset(clef_path, 'clef', mapping['clef'])
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])
    df = pd_concat([df_clef, df_tinman], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_molecular_dataset(clef_path, tinman_path):
    mapping = molecular_mapping()
    df_clef = convert_clef_openi_dataset(clef_path, 'clef', mapping['clef'])
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])
    df = pd_concat([df_clef, df_tinman], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_graphics_dataset(tinman_path, synthetic_path, chart2020_path):
    mapping = graphics_mapping()
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])

    df_synthetic = read_csv(synthetic_path)
    df_synthetic['source'] = 'Chart_Synthetic'

    df_chart2020 = read_csv(chart2020_path)
    df_chart2020['source'] = 'Chart2020'

    df_synthetic['modality'] = df_synthetic['modality'].replace(
        mapping['synthetic'])
    df_chart2020['modality'] = df_chart2020['modality'].replace(
        mapping['chart2020'])

    df_synthetic['caption'] = ''
    df_chart2020['caption'] = ''

    columns = ['img', 'modality', 'source', 'img_path', 'modality', 'caption']
    df = pd_concat([df_synthetic[columns], df_chart2020[columns],
                   df_tinman[columns]], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


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


def experimental_mapping():
    return {
        'clef': {
            'GGEL': 'GEL'
        },
        'tinman': {
            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'GEL',
            'EXPERIMENTAL,GEL,OTHER': 'GEL',
            'EXPERIMENTAL,GEL,RT_PCR': 'GEL',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'GEL',
            'EXPERIMENTAL,PLATE': 'PLATE'
        }
    }


def gel_mapping():
    return {
        'tinman': {
            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'NORTHERN BLOT',
            'EXPERIMENTAL,GEL,OTHER': 'OTHER',
            'EXPERIMENTAL,GEL,RT_PCR': 'RT_PCR',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'WESTERN BLOT'
        }
    }


def microscopy_mapping():
    return {
        'clef': {
            'DMLI': 'LIGHT',
            'DMFL': 'FLUORESCENCE',
            'DMTR': 'ELECTRON',
            'DMEL': 'ELECTRON'
        },
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'FLUORESCENCE',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'FLUORESCENCE',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'FLUORESCENCE',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'FLUORESCENCE',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'FLUORESCENCE',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'LIGHT',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'LIGHT',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'LIGHT',
            'MICROSCOPY,LIGHT,OTHER': 'LIGHT',
            'MICROSCOPY,ELECTRON,OTHER': 'ELECTRON',
            'MICROSCOPY,ELECTRON,SCANNING': 'ELECTRON',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'ELECTRON'
        }
    }


def electron_mapping():
    return {
        'tinman': {
            'MICROSCOPY,ELECTRON,OTHER': 'OTHER',
            'MICROSCOPY,ELECTRON,SCANNING': 'SCANNING',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'TRANSMISSION'
        },
        'clef': {
            'DMEL': 'SCANNING',
            'DMTR': 'TRANSMISSION'
        }
    }


def light_fluorescence_mapping():
    return {
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'INSITU HYBRIDIZATION',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'REPORTER GENES AND IMMUNOHISTOCHEMISTRY',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'EFIC',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'WHOLE MOUNT',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'OTHER',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'INSITU HYBRIDIZATION',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'REPORTER GENES AND IMMUNOHISTOCHEMISTRY',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'WHOLE MOUNT',
            'MICROSCOPY,LIGHT,OTHER': 'OTHER',
        }
    }


def molecular_mapping():
    return {
        'clef': {
            'GCHE': 'CHEMICAL STRUCTURE',
            'GGEN': 'DNA SEQUENCE',
        },
        'tinman': {
            'MOLECULAR STRUCTURE,3D STRUCTURE': '3D STRUCTURE',
            'MOLECULAR STRUCTURE,CHEMICAL STRUCTURE': 'CHEMICAL STRUCTURE',
            'MOLECULAR STRUCTURE,DNA': 'DNA SEQUENCE',
            'MOLECULAR STRUCTURE,PROTEIN': 'PROTEIN SEQUENCE'
        }
    }


def photography_mapping():
    return {
        'clef': {
            'DRDM': 'DERMATOLOGY/SKIN',
            'DVEN': 'ORGANS/BODY PARTS',
            'DVOR': 'ORGANS/BODY PARTS'
        }
    }


def graphics_mapping():
    return {
        'clef': {
            'GFLO': 'FLOWCHART',
            'D3DR': '3D RECONSTRUCTION',
            'DSEM': 'SIGNALS/WAVES',
            'DSEE': 'SIGNALS/WAVES',
            'DSEC': 'SIGNALS/WAVES',
        },
        'tinman': {
            'GRAPHICS,FLOWCHART': 'FLOWCHART',
            'GRAPHICS,HISTOGRAM': 'HISTOGRAM',
            'GRAPHICS,LINECHART': 'LINECHART',
            'GRAPHICS,OTHER': 'OTHER',
            'GRAPHICS,SCATTERPLOT': 'SCATTERPLOT'
        },
        'synthetic': {
            'Scatter': 'SCATTERPLOT',
            'Grouped horizontal bar': 'HISTOGRAM',
            'Grouped vertical bar': 'HISTOGRAM',
            'Horizontal box': 'OTHER',
            'Vertical box': 'OTHER',
            'Stacked horizontal bar': 'HISTOGRAM',
            'Stacked vertical bar': 'HISTOGRAM',
            'Line': 'LINECHART',
            'Pie': 'OTHER',
            'Donut': 'OTHER'
        },
        'chart2020': {
            'area': 'OTHER',
            'heatmap': 'OTHER',
            'horizontal_bar': 'HISTOGRAM',
            'horizontal_interval': 'OTHER',
            'line': 'LINECHART',
            'manhattan': 'OTHER',
            'map': 'OTHER',
            'pie': 'OTHER',
            'scatter': 'SCATTERPLOT',
            'scatter-line': 'SCATTERPLOT',
            'surface': 'OTHER',
            'venn': 'OTHER',
            'vertical_bar': 'HISTOGRAM',
            'vertical_box': 'OTHER',
            'vertical_interval': 'OTHER'
        }
    }


def modality_mapping():
    return {
        'clef': {
            'GFLO': 'GRAPHICS',
            'D3DR': 'GRAPHICS',
            'DSEM': 'GRAPHICS',
            'DSEE': 'GRAPHICS',
            'DSEC': 'GRAPHICS',
            'GFIG': 'GRAPHICS',

            'DRDM': 'PHOTOGRAPHY',
            'DVEN': 'PHOTOGRAPHY',
            'DVOR': 'PHOTOGRAPHY',

            'GCHE': 'MOLECULAR STRUCTURE',
            'GGEN': 'MOLECULAR STRUCTURE',

            'DMEL': 'MICROSCOPY',
            'DMTR': 'MICROSCOPY',
            'DMFL': 'MICROSCOPY',
            'DMLI': 'MICROSCOPY',

            'GGEL': 'EXPERIMENTAL',

            'DRXR': 'RADIOLOGY',
            'DRCT': 'RADIOLOGY',
            'DRUS': 'RADIOLOGY',
            'DRMR': 'RADIOLOGY',
            'DRCO': 'RADIOLOGY',
            'DRPE': 'RADIOLOGY',

            'GTAB': 'OTHER',
            'GPLI': 'OTHER',
            'GSCR': 'OTHER',
            'GNCP': 'OTHER',
            'GSYS': 'OTHER',
            'GMAT': 'OTHER',
            'GHDR': 'OTHER',
        },
        'openi': {
            'DRXR': 'RADIOLOGY',
            'DRCT': 'RADIOLOGY',
            'DRUS': 'RADIOLOGY',
            'DRMR': 'RADIOLOGY'
        },
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'MICROSCOPY',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'MICROSCOPY',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'MICROSCOPY',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'MICROSCOPY',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'MICROSCOPY',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'MICROSCOPY',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'MICROSCOPY',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'MICROSCOPY',
            'MICROSCOPY,LIGHT,OTHER': 'MICROSCOPY',
            'MICROSCOPY,ELECTRON,OTHER': 'MICROSCOPY',
            'MICROSCOPY,ELECTRON,SCANNING': 'MICROSCOPY',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'MICROSCOPY',

            'GRAPHICS,FLOWCHART': 'GRAPHICS',
            'GRAPHICS,HISTOGRAM': 'GRAPHICS',
            'GRAPHICS,LINECHART': 'GRAPHICS',
            'GRAPHICS,OTHER': 'GRAPHICS',
            'GRAPHICS,SCATTERPLOT': 'GRAPHICS',

            'MOLECULAR STRUCTURE,3D STRUCTURE': 'MOLECULAR STRUCTURE',
            'MOLECULAR STRUCTURE,CHEMICAL STRUCTURE': 'MOLECULAR STRUCTURE',
            'MOLECULAR STRUCTURE,DNA': 'MOLECULAR STRUCTURE',
            'MOLECULAR STRUCTURE,PROTEIN': 'MOLECULAR STRUCTURE',

            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'EXPERIMENTAL',
            'EXPERIMENTAL,GEL,OTHER': 'EXPERIMENTAL',
            'EXPERIMENTAL,GEL,RT_PCR': 'EXPERIMENTAL',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'EXPERIMENTAL',
            'EXPERIMENTAL,PLATE': 'EXPERIMENTAL',

            'PHOTOGRAPHY': 'PHOTOGRAPHY',
            'OTHER': 'OTHER'
        }
    }
