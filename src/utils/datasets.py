from pandas import concat as pd_concat
from pandas import read_csv
from os import path, read
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def remove_small_classes(df, target_col, threshold=100):
    gb = df.groupby(target_col)[target_col].count()
    to_remove = []
    for label in gb.index:
        if gb[label] < threshold:
            to_remove.append(label)
    return df[~df.label.isin(to_remove)]

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
    df_gel['label'] = 'exp.gel'
    df_gel['source'] = 'udel-gel'
    df_gel['caption'] = ''

    df_plates = read_csv(plate_csv_path)
    df_plates['img'] = df_plates["filepath"].str.split("/", expand=True)[2]
    df_plates['img_path'] = plate_base_path + df_plates['filepath']
    df_plates['label'] = 'exp.pla'
    df_plates['source'] = 'udel-plate'
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
    df_synthetic['source'] = 'chart-synth'

    df_chart2020 = read_csv(chart2020_path)
    df_chart2020['source'] = 'chart-2020'

    df_synthetic['label'] = df_synthetic['modality'].replace(
        mapping['synthetic'])
    df_chart2020['label'] = df_chart2020['modality'].replace(
        mapping['chart2020'])

    df_synthetic['caption'] = ''
    df_chart2020['caption'] = ''

    columns = ['img', 'source', 'img_path', 'label', 'caption']
    df = pd_concat([df_synthetic[columns], df_chart2020[columns],
                   df_tinman[columns]], axis=0).reset_index(drop=True)
    df = stratify(df)
    return df


def get_high_modality_dataset(clef_path, tinman_path, synthetic_path, chart2020_path, plate_path, gel_path, gel_base_path, plate_base_path):
    mapping = modality_mapping()
    df_tinman = convert_tinman_dataset(tinman_path, mapping['tinman'])
    df_clef = convert_clef_openi_dataset(clef_path, 'clef', mapping['clef'])
    df_openi = convert_clef_openi_dataset(clef_path, 'openi', mapping['openi'])

    df_synthetic = read_csv(synthetic_path)
    df_synthetic['source'] = 'chart-synth'
    df_synthetic['caption'] = ''
    df_synthetic['label'] = 'gra'

    df_chart2020 = read_csv(chart2020_path)
    df_chart2020['source'] = 'chart-2020'
    df_chart2020['caption'] = ''
    df_chart2020['label'] = 'gra'

    df_gel = read_csv(gel_path)
    df_gel['img'] = df_gel["filepath"].str.split("/", expand=True)[2]
    df_gel['img_path'] = gel_base_path + df_gel['filepath']
    df_gel['label'] = 'exp'
    df_gel['source'] = 'udel-gel'
    df_gel['caption'] = ''

    df_plates = read_csv(plate_path)
    df_plates['img'] = df_plates["filepath"].str.split("/", expand=True)[2]
    df_plates['img_path'] = plate_base_path + df_plates['filepath']
    df_plates['label'] = 'exp'
    df_plates['source'] = 'udel-plate'
    df_plates['caption'] = ''

    columns = ['img', 'source', 'img_path', 'label', 'caption']
    df = pd_concat([df_tinman[columns], df_clef[columns],
                   df_openi[columns], df_chart2020[columns], df_synthetic[columns],
                   df_gel[columns], df_plates[columns]], axis=0).reset_index(drop=True)

    df = stratify(df)
    return df


def radiology_mapping():
    return {
        'clef': {
            'DRXR': 'rad.xra',
            'DRUS': 'rad.uls',
            'DRAN': 'rad.ang',
            'DRMR': 'rad.cmp',
            'DRPE': 'rad.cmp',
            'DRCO': 'rad.oth'
        },
        'openi': {
            'DRXR': 'rad.xra',
            'DRCT': 'rad.cmp',
            'DRUS': 'rad.uls',
            'DRMR': 'rad.cmp'
        }
    }


def experimental_mapping():
    return {
        'clef': {
            'GGEL': 'exp.gel'
        },
        'tinman': {
            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'exp.gel',
            'EXPERIMENTAL,GEL,OTHER': 'exp.gel',
            'EXPERIMENTAL,GEL,RT_PCR': 'exp.gel',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'exp.gel',
            'EXPERIMENTAL,PLATE': 'exp.pla'
        }
    }


def gel_mapping():
    return {
        'tinman': {
            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'exp.gel.nor',
            'EXPERIMENTAL,GEL,OTHER': 'exp.gel.oth',
            'EXPERIMENTAL,GEL,RT_PCR': 'exp.gel.rpc',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'exp.gel.wes'
        }
    }


def microscopy_mapping():
    return {
        'clef': {
            'DMLI': 'mic.lig',
            'DMFL': 'mic.flu',
            'DMTR': 'mic.ele',
            'DMEL': 'mic.ele'
        },
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'mic.flu',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic.flu',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'mic.flu',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'mic.flu',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'mic.flu',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'mic.lig',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic.lig',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'mic.lig',
            'MICROSCOPY,LIGHT,OTHER': 'mic.lig',
            'MICROSCOPY,ELECTRON,OTHER': 'mic.ele',
            'MICROSCOPY,ELECTRON,SCANNING': 'mic.ele',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'mic.ele'
        }
    }


def electron_mapping():
    return {
        'tinman': {
            'MICROSCOPY,ELECTRON,OTHER': 'mic.ele.oth',
            'MICROSCOPY,ELECTRON,SCANNING': 'mic.ele.sca',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'mic.ele.tra'
        },
        'clef': {
            'DMEL': 'mic.ele.sca',
            'DMTR': 'mic.ele.tra'
        }
    }


def light_fluorescence_mapping():
    return {
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'mic.flu.ins',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic.flu.rep',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'mic.flu.efi',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'mic.flu.who',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'mic.flu.oth',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'mic.lig.ins',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic.lig.rep',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'mic.lig.who',
            'MICROSCOPY,LIGHT,OTHER': 'mic.lig.oth',
        }
    }


def molecular_mapping():
    return {
        'clef': {
            'GCHE': 'mol.che',
            'GGEN': 'mol.dna',
        },
        'tinman': {
            'MOLECULAR STRUCTURE,3D STRUCTURE': 'mol.3ds',
            'MOLECULAR STRUCTURE,CHEMICAL STRUCTURE': 'mol.che',
            'MOLECULAR STRUCTURE,DNA': 'mol.dna',
            'MOLECULAR STRUCTURE,PROTEIN': 'mol.pro'
        }
    }


def photography_mapping():
    return {
        'clef': {
            'DRDM': 'pho.der',  # DERMATOLOGY/SKIN
            'DVEN': 'pho.obp',  # ORGANS/BODY PARTS
            'DVOR': 'pho.obp'
        }
    }


def graphics_mapping():
    return {
        'clef': {
            'GFLO': 'gra.flow',      # 'FLOWCHART',
            'D3DR': 'gra.3dr',       # '3D RECONSTRUCTION',
            'DSEM': 'gra.sig',       # SIGNALS/WAVES',
            'DSEE': 'gra.sig',
            'DSEC': 'gra.sig',
        },
        'tinman': {
            'GRAPHICS,FLOWCHART': 'gra.flow',  # 'FLOWCHART',
            'GRAPHICS,HISTOGRAM': 'gra.his',  # 'HISTOGRAM',
            'GRAPHICS,LINECHART': 'gra.lin',  # 'LINECHART',
            'GRAPHICS,OTHER': 'gra.oth',  # 'OTHER',
            'GRAPHICS,SCATTERPLOT': 'gra.sca',  # 'SCATTERPLOT'
        },
        'synthetic': {
            'Scatter': 'gra.sca',
            'Grouped horizontal bar': 'gra.his',
            'Grouped vertical bar': 'gra.his',
            'Horizontal box': 'gra.oth',
            'Vertical box': 'gra.oth',
            'Stacked horizontal bar': 'gra.his',
            'Stacked vertical bar': 'gra.his',
            'Line': 'gra.lin',
            'Pie': 'gra.oth',
            'Donut': 'gra.oth'
        },
        'chart2020': {
            'area': 'gra.oth',
            'heatmap': 'gra.oth',
            'horizontal_bar': 'gra.his',
            'horizontal_interval': 'gra.oth',
            'line': 'gra.lin',
            'manhattan': 'gra.oth',
            'map': 'gra.oth',
            'pie': 'gra.oth',
            'scatter': 'gra.sca',
            'scatter-line': 'gra.sca',
            'surface': 'gra.oth',
            'venn': 'gra.oth',
            'vertical_bar': 'gra.his',
            'vertical_box': 'gra.oth',
            'vertical_interval': 'gra.oth'
        }
    }


def modality_mapping():
    return {
        'clef': {
            'GFLO': 'gra',  # GRAPHICS
            'D3DR': 'gra',
            'DSEM': 'gra',
            'DSEE': 'gra',
            'DSEC': 'gra',
            'GFIG': 'gra',

            'DRDM': 'pho',  # PHOTOGRAPHY
            'DVEN': 'pho',
            'DVOR': 'pho',

            'GCHE': 'mol',  # 'MOLECULAR STRUCTURE',
            'GGEN': 'mol',

            'DMEL': 'mic',
            'DMTR': 'mic',
            'DMFL': 'mic',
            'DMLI': 'mic',

            'GGEL': 'exp',

            'DRXR': 'rad',
            'DRCT': 'rad',
            'DRUS': 'rad',
            'DRMR': 'rad',
            'DRCO': 'rad',
            'DRPE': 'rad',

            'GTAB': 'oth',
            'GPLI': 'oth',
            'GSCR': 'oth',
            'GNCP': 'oth',
            'GSYS': 'oth',
            'GMAT': 'oth',
            'GHDR': 'oth',
        },
        'openi': {
            'DRXR': 'rad',
            'DRCT': 'rad',
            'DRUS': 'rad',
            'DRMR': 'rad',
            'MICROSCOPY': 'mic',
            'OTHER': 'oth'
        },
        'tinman': {
            'MICROSCOPY,FLUORESCENCE,INSITU HYBRIDIZATION': 'mic',
            'MICROSCOPY,FLUORESCENCE,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic',
            'MICROSCOPY,FLUORESCENCE,EFIC': 'mic',
            'MICROSCOPY,FLUORESCENCE,WHOLE MOUNT': 'mic',
            'MICROSCOPY,FLUORESCENCE,OTHER': 'mic',
            'MICROSCOPY,LIGHT,INSITU HYBRIDIZATION': 'mic',
            'MICROSCOPY,LIGHT,REPORTER GENES AND IMMUNOHISTOCHEMISTRY': 'mic',
            'MICROSCOPY,LIGHT,WHOLE MOUNT': 'mic',
            'MICROSCOPY,LIGHT,OTHER': 'mic',
            'MICROSCOPY,ELECTRON,OTHER': 'mic',
            'MICROSCOPY,ELECTRON,SCANNING': 'mic',
            'MICROSCOPY,ELECTRON,TRANSMISSION': 'mic',

            'GRAPHICS,FLOWCHART': 'gra',
            'GRAPHICS,HISTOGRAM': 'gra',
            'GRAPHICS,LINECHART': 'gra',
            'GRAPHICS,OTHER': 'gra',
            'GRAPHICS,SCATTERPLOT': 'gra',

            'MOLECULAR STRUCTURE,3D STRUCTURE': 'mol',
            'MOLECULAR STRUCTURE,CHEMICAL STRUCTURE': 'mol',
            'MOLECULAR STRUCTURE,DNA': 'mol',
            'MOLECULAR STRUCTURE,PROTEIN': 'mol',

            'EXPERIMENTAL,GEL,NORTHERN BLOT': 'exp',
            'EXPERIMENTAL,GEL,OTHER': 'exp',
            'EXPERIMENTAL,GEL,RT_PCR': 'exp',
            'EXPERIMENTAL,GEL,WESTERN BLOT': 'exp',
            'EXPERIMENTAL,PLATE': 'exp',

            'PHOTOGRAPHY': 'pho',
            'OTHER': 'oth'
        }
    }
