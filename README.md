# Lenders Segmentation at Kiva.org

## Motivation

### Who is Kiva.org?
Kiva is an international nonprofit, founded in 2005 and based in San Francisco with a mission to connect people through lending to alleviate poverty. Until today, there are more than $1.2B loans funded through Kiva, and there are 1.7M active lenders on the Kiva website.

### Why Lenders Segmentation?
Understanding the lenders' community is crucial to Kiva. People lent through Kiva with various motivations. Being able to segment lenders based on their identities and behaviors can help Kiva to personalize their service to different lender group.

## Data and Key Findings

### Data Profiles
The data is from private Kiva data. The dataset contains ~3 million registered users and 109 features including the identity, the locations, the purchasing, donation and deposit behaviors, and the loan preferences.

## Unsurpervised Machine Learning

From  Explorative Data Analysis, I identified normal lenders from super lenders and potential lenders based on their purchasing, donating and depositing behaviors. Later I used the Principle Component Analysis on normal lenders group to reduce the dimensionality from 66 features to 6 features while retaining 50% of the information. Lastly, I used Kmeans++ unsupervised machine learning algorithms to cluster normal users into six distinct clusters. The cluster number is chosen based on the silhouette score. Here we are using 3 dimensionalities as a good proxy for 6 dimensionalities for visualization purpose.

## Visualization

## Future work
 - Potential Lenders: what are stopping them?

- Normal Lenders - the under performing groups:
    - conversion rate 
    - Engagement

- Super Users: how can we best support them?

