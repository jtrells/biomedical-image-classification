import numpy as np
from cuml.manifold import TSNE as cumlTSNE
from cuml import UMAP as cumlUMAP
from cuml import PCA as cumlPCA
from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors
from sklearn.preprocessing import LabelEncoder

def reduce_dimensions(df, reducer_name, subset, subset_col, num_dimensions=2, add_hits=True, label_col='label'):
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
    embeddings = reducer.fit_transform(features)

    if (add_hits):
        n_hits = calc_neighborhood_hit(df, embeddings, n_neighbors=6, label_col=label_col)
        return embeddings, n_hits

    return embeddings, None

def calc_neighborhood_hit(df, embeddings, n_neighbors=6, label_col='label'):    
    neigh = cumlNearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(embeddings)
    le = LabelEncoder().fit(df[label_col].unique())
    n_hits = []
    for neighborhood in neigh.kneighbors(embeddings, n_neighbors + 1, return_distance=False):
        labels  = le.transform(df.iloc[neighborhood][label_col].values)
        targets = [labels[0]] * (len(labels) - 1) 
        n_hit = np.mean(targets == labels[1:])
        n_hits.append(n_hit)
    return n_hits