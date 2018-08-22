from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import locally_linear_embedding
from time import time


def PCA_reduce(X,dimensionality):
    '''return reduced n dimensionality matrix and np array of important features'''
    important_features = []
    pcd = PCA(dimensionality).fit(X)
    pca_matrix = pcd.components_
    for i in range(dimensionality):
        print('#{} Principle Component top 5 contributors are:'.format(i), np.argsort(np.abs(pca_matrix[i,:]))[-5:])
        important_features.append(np.argsort(np.abs(pca_matrix[i,:]))[-5:])
    X_reduced = np.dot(X,pcd.components_.T)
    print('Reduced X shape:', X_reduced.shape)
    important_features = np.array(important_features)
    print('Explained variance:', pcd.explained_variance_)
    print('Explained variance ratio:', pcd.explained_variance_ratio_)
    print('Explained variance ratio accumulated:', pcd.explained_variance_ratio_.cumsum())
    return X_reduced, important_features

def print_imp_features(df,imp_features):
    '''print important features names and return the counter of the features'''
    feature = []
    idx = 0
    c = 1
    while idx < len(np.array(imp_features).flatten()):
        print('#{} PC:'.format(c))
        if idx+5 > len(np.array(imp_features).flatten()):
            for i in np.array(imp_features).flatten()[idx:]:
                feature.append(list(df.columns)[i])
                print(list(df.columns)[i])
        else:
            for i in np.array(imp_features).flatten()[idx:idx+5]:
                feature.append(list(df.columns)[i])        
                print(list(df.columns)[i])
        idx += 5
        c +=1
    counter = Counter(feature)    
    return counter


def kmeans_cluster(X_reduced,cluster_num):
    '''return y (assigned_cluster for each row) and centers of the clusters '''
    kmeans = KMeans(init='k-means++', n_clusters=cluster_num, n_init=10,tol=0.0001,verbose=0)
    y = kmeans.fit_predict(X_reduced)
    centers = kmeans.cluster_centers_
    # assigned_cluster = kmeans.transform(X_reduced).argmin(axis=1)
    return y, centers

# def drop_outliers(df,outlier_index_lst=[2987,27627,15038,19433,704]):
#     '''drop the outliers identifed from PCA and Kmeans'''
#     df = df.drop(df.index[outlier_index_lst])
#     return df


def make_dendrogram(X, linkage_method, metric, figsize=(25,15),color_threshold=None):
    '''
    This function creates and plots the dendrogram created by hierarchical clustering.
    
    INPUTS: Pandas Dataframe, string, string, int
    
    OUTPUTS: None
    '''
    distxy = squareform(pdist(X, metric=metric))
    Z = linkage(distxy, linkage_method)
    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
#         labels = dataframe.index,
        color_threshold = color_threshold
    )
    plt.show()


def DBSCAN_cluster(X,eps=1,min_samples=10):
    '''return labels and n_clusters'''
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return labels,n_clusters_

def lle_dimensionality_reduction(X, n_neighbors, n_dimensionality):
    time0=time()
    X_lle,err = locally_linear_embedding(X,n_neighbors,n_dimensionality)
    time1=time()
    print(time1-time0)
    return X_lle,err
