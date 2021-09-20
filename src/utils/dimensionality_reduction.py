import numpy as np
from cuml.manifold import TSNE as cumlTSNE
from cuml import UMAP as cumlUMAP
from cuml import PCA as cumlPCA
from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors

def reduce_dimensions(features, reducer_name, num_dimensions=2):
    """ Reduce the dimensionality of the provided features

    Attributes
    ----------
    features: 2D np.array with one N-feature vector per row
    num_dimensions: dimensions of output vectors

    """
    if reducer_name == 'tsne':
        reducer = cumlTSNE(n_components=num_dimensions, method='barnes_hut')        
    elif reducer_name == 'umap':
        reducer = cumlUMAP(n_neighbors=15, n_components=num_dimensions, n_epochs=500, min_dist=0.1)
    elif reducer_name == 'pca':
        reducer = cumlPCA(n_components = 2)
    else:
        return None
    embeddings = reducer.fit_transform(features)

    return embeddings

def calc_neighborhood_hit(df, embeddings, le, n_neighbors=6, label_col='label'):
    """ Calculate the neighborhood hit for each data point.

    Data points with low scores have a considerable proportion of neighbors with 
    different predicted labels.

    Attributes
    -----------
    df: pandas dataframe
    embeddings: 2D numpy.ndarray
    le: label encoder
    """ 
    neigh = cumlNearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(embeddings)

    n_hits = []
    for neighborhood in neigh.kneighbors(embeddings, n_neighbors + 1, return_distance=False):
        labels  = le.transform(df.iloc[neighborhood][label_col].values)
        targets = [labels[0]] * (len(labels) - 1) 
        n_hit = np.mean(targets == labels[1:])
        n_hits.append(n_hit)
    return n_hits


def get_neighbors_by_index(df, indexItem, n_neighbors=10):
    features = np.vstack(df.features.values)
    total_neighbors = n_neighbors + 1   # because model returns item itself
    model = cumlNearestNeighbors(n_neighbors=total_neighbors, algorithm='brute').fit(features)
    itemFeatures = np.vstack([df.iloc[indexItem].features])
    distances, indices = model.kneighbors(itemFeatures)
    # first item is the queried item itself
    distances = distances[0][1:]
    indices = indices[0][1:]

    output_df = df.filter(items = indices, axis=0)
    output_df.loc[:,'distances'] = distances

    return output_df

