import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

REDUCED_EMBEDDINGS_PATH = 'data/reduced_embeddings.npy'
CLUSTER_PATH = 'data/cluster_labels.npy'
K = 8

def reduce_embeddings(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    np.save(REDUCED_EMBEDDINGS_PATH, reduced_embeddings)
    return reduced_embeddings

def perform_clustering(reduced_embeddings):
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(reduced_embeddings)
    np.save(CLUSTER_PATH, labels)
    return labels

def load_cluster_labels():
    if os.path.exists(CLUSTER_PATH):
        return np.load(CLUSTER_PATH)
    else:
        return None
