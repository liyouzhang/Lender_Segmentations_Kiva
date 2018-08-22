from sklearn.decomposition import PCA
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
    '''[summary]
    reduced n dimensionality matrix and np array of important features

    Arguments:
        X {[numpy matrix]} -- input matrix
        dimensionality {[int]} -- to what dimensionality to reduce to

    Returns:
        X_reduced -- reduced matrix
        important_features -- a list of features that contributed most to the principle component
    '''

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
    '''[summary]
    print important features names and return the counter of the features

    Arguments:
        df {[pandas dataframe]} -- input
        imp_features {[list]} -- important features returned from pca_reduce function

    Returns:
        counter [dictionary] -- a dictionary with feature name as key, and frequency showed as value 
    '''

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
    '''[summary]
    use kmeans++ algorithems to cluster input data.

    Arguments:
        X_reduced {[numpy matrix]} -- input matrix
        cluster_num {[int]} -- how many clusters for clustering 

    Returns:
        y[numpy array] -- assigned_cluster for each row
        centers[numpy matrix] -- centers of the clusters
    '''

    kmeans = KMeans(init='k-means++', n_clusters=cluster_num, n_init=10,tol=0.0001,verbose=0)
    y = kmeans.fit_predict(X_reduced)
    centers = kmeans.cluster_centers_
    # assigned_cluster = kmeans.transform(X_reduced).argmin(axis=1)
    return y, centers

