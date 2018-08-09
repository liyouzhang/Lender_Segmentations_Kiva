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
    return X_reduced, important_features

def print_imp_features(df,imp_features):
    for i in np.array(imp_features).flatten():
        print(list(df.columns)[i])
    print ("mode:", stats.mode(np.array(imp_features).flatten()))

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
    '''return y and assigned_cluster for each row'''
    kmeans = KMeans(init='k-means++', n_clusters=cluster_num, n_init=10,tol=0.01,verbose=0)
    y = kmeans.fit_predict(X_reduced)
    assigned_cluster = kmeans.transform(X_reduced).argmin(axis=1)
    return y, assigned_cluster

def plot_2D_kmeans(X_reduced,y,xlim_left,xlim_right,ylim_down,ylim_up):
    '''plot for kmeans results, adjust ax lim to zoom in/out'''
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
    ax.set_xlim(xlim_left,xlim_right)
    ax.set_ylim(ylim_down,ylim_up)
    ax.set_title("Scatterplot in PCA 2-Plane with clustering results")
    plt.show()


def plot_3D_kmeans(X_reduced,y):
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
    plt.show()