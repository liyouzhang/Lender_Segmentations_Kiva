# Personalized_loan_recommendations


## Recommendation systems

### Data:
- Implicit feedback


### Lenders Clustering

- purpose:
    - to discover latent features
    - EDA

- Kmeans
    - assign points to centroids based on distance

### Recommendation system:
 
- Matrix Factorization:

    - find topic

- Collabrative filtering:  
consider past users behaviors and recommende based on how other users interact with other items (loans)

    - user-user similarities
    - item-item similarities

    - how to define similarities?
        - distance
        - correlation
        - cosine similarity
        - Jaccord Index

- content-based:
recommend base on the characteristics of loans and users behaviors are not considered

- popularity -- for new users:  
recommend base on the popularity of loans. Same recommendations for all users.


### Measurement
- cross validation
    - holding the latest loans as the validation set
