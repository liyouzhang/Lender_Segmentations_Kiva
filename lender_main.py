from lender_pipeline import pipeline

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.style.use('ggplot')
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler



ls = pd.read_csv('lender_dataset_sampled.csv')
X = pipeline(ls)
print(X.shape)