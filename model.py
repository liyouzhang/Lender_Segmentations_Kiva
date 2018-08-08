## 2. Dimensionality reduction

### 2.1 Why reduce dimensionality?
# - 105 columns of features, many of them are correlated to each other
# - if keep all the features, we will overfit our model and add on a lot of noise
# - reduce dimensionality helps us overcome the curse of dimensionality in clustering methods

### 2.2 PCA (Principle Components Analysis)


# minidf.dtypes
# minidf.isnull().sum()

list(ls.columns)[18]

list(ls.columns)[80]

list(ls.columns)[58]

list(ls.columns)[87]

list(ls.columns)[83]

list(ls.columns)[14]

from sklearn.decomposition import PCA, TruncatedSVD

def PCA_reduce(X):
    pcd = PCA(2).fit(X)
    pca_matrix = pcd.components_
    print('First Principle Component main contributors:', np.argsort(np.abs(pca_matrix[0,:]))[-3:]) # first PCA)
    print('Second Principle Component main contributors:',np.argsort(np.abs(pca_matrix[1,:]))[-3:])# second PCA
    X_reduced = np.dot(X,pcd.components_.T)
    print('Reduced X shape:', X_reduced.shape)
    return X_reduced

def 
# X_reduced.shape

# fig, ax = plt.subplots(1, figsize=(8, 6))

# ax.scatter(X_reduced[:, 0], X_reduced[:, 1]);
# ax.set_title("Scatterplot in PCA 2-Plane")
# ax.set_xlabel("First Principal Component")
# ax.set_ylabel("Second Principal Component")

# X_reduced_df = pd.DataFrame(X_reduced)

# normal_users = X_reduced_df[X_reduced_df.iloc[:,0]<500000][X_reduced_df.iloc[:,1]<200000]

fig, ax = plt.subplots(1, figsize=(8, 6))

ax.scatter(normal_users.values[:, 0], normal_users.values[:, 1]);
ax.set_title("Scatterplot in PCA 2-Plane")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")

# It shows that we have some outliers that deviate from the normal users group, which is in accordance with our knowledge of 'super users'.

### 2.3 Seperate outliers from normal users

# first principal component are mostly composed by below features
minidf.iloc[:,[17, 18,  6, 14,  7]] 

minidf.iloc[:,[17, 18,  6, 14,  7]].describe()

notice:
- high std
- extreme max values

choose top 20% as outliers group

quantiles = minidf.iloc[:,[17, 18,  6, 14,  7]].quantile(0.8)

minidf['outliers?'] = (minidf.iloc[:,7] >= minidf.iloc[:,7].quantile(0.8)) | (minidf.iloc[:,6] >= minidf.iloc[:,6].quantile(0.8)) | (minidf.iloc[:,14] >= minidf.iloc[:,14].quantile(0.8))

outliers = minidf[minidf['outliers?']]

outliers.shape[0]/minidf.shape[0]

churners = outliers[outliers['lifetime_ave_tip_rate'] == 0 ]

churners[['LIFETIME_ACCOUNT_LOAN_PURCHASE_NUM','LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL','CORE_LOAN_PURCHASE_NUM','CORE_LOAN_PURCHASE_TOTAL','DIRECT_LOAN_PURCHASE_NUM','DIRECT_LOAN_PURCHASE_TOTAL','NUM_TEAM_MEMBERSHIPS']].describe()

outliers[outliers['lifetime_ave_tip_rate'] == 0 ].T

pd.scatter_matrix(outliers[['LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL','CORE_LOAN_PURCHASE_TOTAL','LIFETIME_DEPOSIT_TOTAL']],figsize=(10,16));

minidf[~minidf['outliers?']].shape



minidf[minidf['outlier?']].CORE_LOAN_PURCHASE_TOTAL

outliers = minidf[minidf.iloc[:,7] >= minidf.iloc[:,7].quantile(0.8)].shape

def filter_outliers(df, col_index, pct):
    df['outli'[df.iloc[:,col_index] < df.iloc[:,col_index].quantile(pct)]
    outliers = df[~(df.iloc[:,col_index] < df.iloc[:,col_index].quantile(pct))]
    return df, outliers
    # minidf[minidf.iloc[:,7] < minidf.iloc[:,7].quantile(0.8)]

minidf.shape

minidf, outliers = filter_outliers(minidf,7,0.8)

minidf.shape

outliers.shape

minidf.shape[0] + outliers.shape[0]

filter_outliers(minidf,14,0.8).shape



## 3. Clustering

### 3.1 KMeans ++

from sklearn.cluster import KMeans 

import scipy.stats as scs

kmeans = KMeans(init='k-means++', n_clusters=15, n_init=10,tol=0.01,verbose=0)



y = kmeans.fit_predict(X_reduced)

silhouette_score(X_reduced,y)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
ax.set_xlim(-2,2)
ax.set_ylim(-5,5)

kmeans.cluster_centers_

majority = X_reduced_df[X_reduced_df.iloc[:,0]<150000][X_reduced_df.iloc[:,1]<50000]

majority.shape

y1 = kmeans.fit_predict(majority.values)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(majority.values[:,0],majority.values[:,1],c=y1,linewidths=0)

X.shape[0] - majority.shape[0] # only 20 outliers lol....

maj1 = majority[majority.iloc[:,0]<30000][majority.iloc[:,1]<10000] #~120 outliers

maj1.shape

ym1 = kmeans.fit_predict(maj1.values)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(maj1.values[:,0],maj1.values[:,1],c=ym1,linewidths=0)
# ax.set_xlim(15000,18000)

maj1 = majority[majority.iloc[:,0]<30000][majority.iloc[:,1]<10000] #~120 outliers

maj2 = maj1[maj1.iloc[:,0]<15750][maj1.iloc]

X.shape[0] - maj2.shape[0] # 311 outliers

ym2 = kmeans.fit_predict(maj2.values)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(maj2.values[:,0],maj2.values[:,1],c=ym2,linewidths=0);
# ax.set_xlim(15000,18000)

assigned_cluster = kmeans.transform(maj2.values).argmin(axis=1)

maj2['cluster'] = assigned_cluster

maj2[maj2['cluster'] == 0].shape

maj2[maj2['cluster'] == 1].shape

maj3 = maj2[maj2['cluster'] == 0]

ym3 = kmeans.fit_predict(maj3.values)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(maj3.values[:,0],maj3.values[:,1],c=ym3,linewidths=0);

**convert back cluster number**

kmeans = KMeans(init='k-means++', n_clusters=20, n_init=10,tol=0.01,verbose=0)
y = kmeans.fit_predict(X_reduced)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
ax.set_xlim(-5,200)
ax.set_ylim(-110,10)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
ax.set_xlim(-5,25)
ax.set_ylim(-40,10)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
ax.set_xlim(-2,5)
ax.set_ylim(-20,5)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_reduced[:,0],X_reduced[:,1],c=y,linewidths=0)
ax.set_xlim(-1,1)
ax.set_ylim(-5,5)

assigned_cluster = kmeans.transform(X_reduced).argmin(axis=1)

assigned_cluster

X_reduced_df['cluster'] = assigned_cluster

X_reduced_df.groupby('cluster').agg({0:['mean','count'],1:'mean'})

ls['cluster'] = X_reduced_df['cluster']

ls.head()

for i in list(ls[ls['cluster'] == 1].columns):
    print(i, ls.loc[27627,i])

cluster14.describe().loc[['mean','min','25%','50%','75%','max'],:].T

cluster14[['LIFETIME_DEPOSIT_TOTAL',]]

#### 3.1.2 choosing k

from sklearn.metrics import silhouette_score, silhouette_samples

import itertools

x = X_reduced

silhouette_score(x,y)

maxk = 8
wcss = np.zeros(maxk)
silhouette = np.zeros(maxk)

fig, axes = plt.subplots(3, 4, figsize=(16,9))

# flatten
axes = [ax for axrow in axes for ax in axrow]

for k, ax in zip(range(1,maxk), axes):
    km = KMeans(k)
    y = km.fit_predict(x)
    ax.axis('off')
    ax.scatter(x[:,0], x[:,1], c=y, linewidths=0, s=10)
    ax.set_ylim(ymin=-9, ymax=8)
    
    
    for c in range(0, k):
        for i1, i2 in itertools.combinations([ i for i in range(len(y)) if y[i] == c ], 2):
            wcss[k] += sum(x[i1] - x[i2])**2
    wcss[k] /= 2
    
    if k > 1:
        silhouette[k] = silhouette_score(x,y)

1. FE - non linear relationships - PCA can reduce linear combinations
2. cluster on a few features first - business intuition
3. can I use Gini Punity -- if I can create labels? (e.g. donations)

### 3.2 HCA

def make_dendrogram(dataframe, linkage_method, metric, color_threshold=None):
    '''
    This function creates and plots the dendrogram created by hierarchical clustering.
    
    INPUTS: Pandas Dataframe, string, string, int
    
    OUTPUTS: None
    '''
    distxy = squareform(pdist(dataframe.values, metric=metric))
    Z = linkage(distxy, linkage_method)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
        labels = dataframe.index,
        color_threshold = color_threshold
    )
    plt.show()

### 3.3 NMF

DBSCAN  -- good for outliers