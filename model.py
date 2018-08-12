from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly
plotly.tools.set_credentials_file(username='liyouzhang', api_key='gSJMts7w7BogVSyqxiMq')
import plotly.plotly as py
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import locally_linear_embedding
from time import time
from math import pi

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
    print('1st PC:')
    for i in np.array(imp_features).flatten()[:5]:
        feature.append(list(df.columns)[i])        
        print(list(df.columns)[i])
    print('2nd PC:')
    for i in np.array(imp_features).flatten()[5:10]:
        feature.append(list(df.columns)[i])        
        print(list(df.columns)[i])
    print('3rd PC:')
    for i in np.array(imp_features).flatten()[10:]:
        feature.append(list(df.columns)[i])        
        print(list(df.columns)[i])
    counter = Counter(feature)    
    return counter

def plot_2D_reduced_X(X_reduced):
    '''plot for 2D - PCA'''
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1]);
    ax.set_title("Scatterplot in PCA 2-Plane")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    plt.show()

def plotly_3D_reduced_X(X_reduced):
    '''use plotly to visualize the 3D reduced X'''
    trace1 = go.Scatter3d(
    x=X_reduced[:,0],
    y=X_reduced[:,1],
    z=X_reduced[:,2],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
            ),
        opacity=0.8
        )
    )
    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')

def kmeans_cluster(X_reduced,cluster_num):
    '''return y (assigned_cluster) for each row'''
    kmeans = KMeans(init='k-means++', n_clusters=cluster_num, n_init=10,tol=0.0001,verbose=0)
    y = kmeans.fit_predict(X_reduced)
    # assigned_cluster = kmeans.transform(X_reduced).argmin(axis=1)
    return y

def plot_2D_kmeans(X_reduced,y,xlim_left,xlim_right,ylim_down,ylim_up):
    '''plot for kmeans results, adjust ax lim to zoom in/out'''
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
    ax.set_xlim(xlim_left,xlim_right)
    ax.set_ylim(ylim_down,ylim_up)
    ax.set_title("Scatterplot in PCA 2-Plane with clustering results")
    plt.show()


def plot_3D_kmeans(X_reduced,y,xlim=None,ylim=None,zlim=None):
    '''use matplotlib to plot the 3D kmeans cluster results'''
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # for c, m in [('r', 'o'), ('b', '^')]:
    xs=X_reduced[:,0]
    ys=X_reduced[:,1]
    zs=X_reduced[:,2]
    ax.scatter(xs, ys, zs, c=y, marker='^')
    ax.set_xlabel('Principal Component One')
    ax.set_ylabel('Principal Component Two')
    ax.set_zlabel('Principal Component Three')
    ax.set_title("Scatterplot in PCA 3-Plane with clustering results")
    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim != None:
        ax.set_ylim(ylim[0],ylim[1])
    if zlim != None:
        ax.set_zlim(zlim[0],zlim[1])
    plt.show()

def matplotlib_3D_X_reduced(X_reduced,label1="First Principle Component",label2="Second Principle Component",label3="Third Principle Component",title="Scatterplot in PCA 3-Plane with clustering results"):
    '''use matplotlib to plot the 3D PCA results'''
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # for c, m in [('r', 'o'), ('b', '^')]:
    xs=X_reduced[:,0]
    ys=X_reduced[:,1]
    zs=X_reduced[:,2]
    ax.scatter(xs, ys, zs, c='r', marker='^')
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(label3)
    ax.set_title(title)
    plt.show()

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

def plot_radar(df, dpi = 64, category=False,num_of_cat=False):
    '''plot spider graph to interpret clustering results
    INPUT - df: cluster number as index'''
    # initialize the figure
    my_dpi=dpi
    plt.figure(figsize=(800/my_dpi, 1133/my_dpi), dpi=my_dpi)
    plt.tight_layout()
    
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))

    for row in range(0, len(df.index)):
        make_spider(df=df, row=row, title='group{}'.format(row), color=my_palette(row),category=category,num_of_cat=num_of_cat)

def make_spider(df, row, title, color,category,num_of_cat):
    '''plot spider graph to interpret clustering results. called in plot_radar function'''
    # number of variable
    categories=list(df)
    if category == True:
        cat =[]
        for c in categories:
            cat.append(c.split("_")[-1])
        categories = cat
    elif num_of_cat == True:
        cat =[]
        for c in categories:
            cat.append(" ".join(c.split("_")[2:-1]))
        categories = cat
    
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(3,2,row+1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    #plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    #plt.ylim(0,40)

    # Ind1
    values=df.loc[row].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)