import numpy as np
from cuml.manifold import TSNE as cumlTSNE
from cuml import UMAP as cumlUMAP
from cuml import PCA as cumlPCA

def reduce_dimensions(df, reducer_name, subset, subset_col, num_dimensions=2):
    if subset != None:
        df = df[df[subset_col]==subset].reset_index(drop = True)
    features = np.vstack(df.features.values)

    if reducer_name == 'tsne':
        reducer = cumlTSNE(n_components=num_dimensions, method='barnes_hut')        
    elif reducer_name == 'umap':
        reducer = cumlUMAP(n_neighbors=15, n_components=num_dimensions, n_epochs=500, min_dist=0.1)
    elif reducer_name == 'pca':
        reducer = cumlPCA(n_components = 2)
    else:
        return None
    embedding = reducer.fit_transform(features)

    return embedding