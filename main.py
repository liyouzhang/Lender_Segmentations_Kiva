import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
%matplotlib inline

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# lenders = pd.read_csv('kiva_ds_csv/lenders.csv')

# loans = pd.read_csv('kiva_ds_csv/loans.csv', parse_dates=['POSTED_TIME', 'PLANNED_EXPIRATION_TIME', 'DISBURSE_TIME',
#        'RAISED_TIME'])

loans_lenders = pd.read_csv('kiva_ds_csv/loans_lenders.csv')
loans_lenders["LENDERS_lst"] = loans_lenders["LENDERS"].map(lambda x: x.split(','))
loans_lenders_drop = loans_lenders.drop('LENDERS',axis=1)

lst_col = 'LENDERS_lst'
df_ll = pd.DataFrame({col:np.repeat(loans_lenders[col].values, loans_lenders[lst_col].str.len())for col in loans_lenders.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(loans_lenders[lst_col].values)})[loans_lenders.columns.tolist()]
#strip extra space and convert to lower case
df_ll["LENDERS_lst"] = df_ll.LENDERS_lst.apply(lambda x: x.strip().lower()) 

descriptions = loans_lenders["LENDERS"]
vect = CountVectorizer()
model = vect.fit_transform(descriptions)

# def tokenize(doc):
#     translater = str.maketrans('','',string.punctuation)
#     a = [word_tokenize(descriptions.translate(translater))]
#     return a

# vect = TfidfVectorizer(analyzer='word',tokenizer=tokenize)
# model = vect.fit_transform(descriptions)
# features = vect.get_feature_names()

vocs = vect.vocabulary_

df_ll['lender_v'] = df_ll['LENDERS_lst'].map(lambda x: vocs[x])


df_l = df_ll.copy()
df_l = df_l.drop('LENDERS_lst',axis = 1)
df_array = df_l.values
df_sparse = csr_matrix( (np.ones(df_array.shape[0]), (df_array[:,0], df_array[:,1])), shape=(1600000,1600000))


u,sigma,vt = svds(df_sparse,k=25)
sigma = np.diag(sigma)
temp = np.dot(u,sigma)
pred = np.dot(temp,vt)