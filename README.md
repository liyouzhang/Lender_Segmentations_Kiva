# Lenders Segmentation at Kiva.org

[Presentation Link](https://docs.google.com/presentation/d/e/2PACX-1vTGus_OgudE6i2v9zcD-lBMmpm-x1KvW-vm2aWhnBZq8nOLBbJDvF_ZTIoM_Z68jMsoKUb7y9liV29f/pub?start=true&loop=false&delayms=3000)



## Motivation

### Who is Kiva.org?
Kiva is an international nonprofit, founded in 2005 and based in San Francisco with a mission to connect people through lending to alleviate poverty. Until today, there are more than $1.2B loans funded through Kiva, and there are 1.7M active lenders on the Kiva website.

![Kiva](https://github.com/liyouzhang/Lender_Segmentations_Kiva/blob/master/pics/kiva.jpg "slide from presentation")

### Why Lenders Segmentation?
Understanding the lenders' community is crucial to Kiva. People lent through Kiva with various motivations. Being able to segment lenders based on their identities and behaviors can help Kiva to personalize their service to different lender groups.

## Data and Algorithms

The Jupyter Notebook walks through the process to produce the results.

### Data Profiles
The data is from private Kiva data. The dataset contains ~3 million registered users and 109 features including the identity, the locations, the purchasing, donation and deposit behaviors, and the loan preferences.

![Data Profile](https://github.com/liyouzhang/Lender_Segmentations_Kiva/blob/master/pics/data_profiles.jpg "slide from presentation")


### Dimensionality Reduction with PCA & Clustering with KMeans++

From  Explorative Data Analysis, I identified normal lenders from super lenders and potential lenders based on their purchasing, donating and depositing behaviors. Later I used the Principle Component Analysis on normal lenders group to reduce the dimensionality from 66 features to 6 features while retaining 50% of the information. Lastly, I used Kmeans++ unsupervised machine learning algorithms to cluster normal users into six distinct clusters. The cluster number is chosen based on the silhouette score. Here we are using 3 dimensionalities as a good proxy for 6 dimensionalities for visualization purpose.

![PCA](https://github.com/liyouzhang/Lender_Segmentations_Kiva/blob/master/pics/PCA.jpg "slide from presentation")

![Kmeans++](https://github.com/liyouzhang/Lender_Segmentations_Kiva/blob/master/pics/clustering.jpg "slide from presentation")


## Recommendations

1. **Churn prevention**  
Engagement at each group’s critical point

2. **Effective Marketing**  
Content: educate lenders based on their behavior patterns and business priorities:
     -  Conversion rate
     -  Donation amount
     -  … …

## Future directions
 - Potential Lenders: what are blockers for them from their first purchase?

- Super Lenders: how can we keep super lenders engaged?


