# Personalized_loan_recommendations


## Recommendation systems

### Data:
- Implicit feedback

### Approach:

- Matrix Factorization:


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
