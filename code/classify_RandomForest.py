
# coding: utf-8

# In[1]:

#get_ipython().magic(u'load_ext watermark')

#get_ipython().magic(u"watermark -a 'Vahid Mirjalili' -d -p scikit-learn,numpy,numexpr,pandas,matplotlib,plotly -v")


# In[2]:

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import scipy
import sklearn

#get_ipython().magic(u'matplotlib inline')


# ## 1. Read the training and test dataset

# In[3]:

df = pd.read_table('data/labeledTrainData.tsv')

print(df.head())


# In[4]:

df_test = pd.read_table('data/testData.tsv')

print(df_test.head())


# ### 1.1 Extracting X & y data columns

# In[5]:

data_train = df.loc[:, 'review']

y_train = df.loc[:, 'sentiment']

print(data_train.head())


# In[6]:

data_test = df_test.loc[:, 'review']

print(data_test.tail())


# ## 2. Text Feature Extraction

# In[7]:

import nltk
import string
import re
from collections import Counter

from nltk.corpus import stopwords


# ### 2.1 Tokenizer Function

#  **Transform to lower-case**  
#  **Remove the punctuations**  
#  **Remove the stopwrods**  
#  **Tokenize the remaining string**  

# In[8]:

## For more info, see http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

stemmer = nltk.stem.porter.PorterStemmer()

def get_tokens(inp_txt):
    
    ## Lower case: ABC -> abc
    txt_lower = inp_txt.lower()
  
    ## Remove punctuations (!, ', ", ., :, ;, )
    #txt_lower_nopunct = txt_lower.translate(string.maketrans("",""), string.punctuation)
    #print(txt_lower_nopunct)
    
    
    ## Tokenize:
    tokens = nltk.word_tokenize(txt_lower) #_nopunct)
    #tokens = nltk.wordpunct_tokenize(txt_lower)
    
    ## remove stop-words:
    tokens_filtered = [w for w in tokens if not w in stopwords.words('english')]
    
    ## stemming:
    stems = [stemmer.stem(t) for t in tokens_filtered]
    stems_nopunct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]
    return (stems_nopunct)


# ### 2.2 TF-IDF Feature Extraction

# 

# ## 3. Apply Random Forest
# 
# 
# Using **sklearn.ensemble.RandomForestClassifier** with the tunable paramaters:
# 
#   * **n_estimators**: number of trees
#   * **criterion:** 'gini', 'entropy'
# 
# **Important Note:** This module, requires dense matrix as input. If a sparse matrix is given, the following error will be raised:  
#    *A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.*

# In[9]:

## For the purpose of represntation
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline
from mlxtend.sklearn import DenseTransformer

tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
    encoding = 'utf-8',
    decode_error = 'replace',
    strip_accents = 'ascii',
    analyzer = 'word',
    max_features = 100,
    smooth_idf = True,
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english',
    tokenizer = get_tokens
)

clf_pipe = pipeline.Pipeline([
    ('vect', tfidf),
    ('densify', DenseTransformer()),
    ('clf', RandomForestClassifier(n_estimators = 10, criterion='gini'))
])



rf_model = clf_pipe.fit(data_train[0:1000], y_train[0:1000])

pred_rf = rf_model.predict(data_test[0:1000])

pred_ef = np.vstack((df_test.loc[0:999, 'id'], pred_rf)).T

print(pred_rf.shape)


# ### Applying Random Forest to All Data

# ## 3. Hyper-parameter Optimization using KFold Cross Validation

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline
from mlxtend.sklearn import DenseTransformer
import datetime

tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
    encoding = 'utf-8',
    decode_error = 'replace',
    strip_accents = 'ascii',
    analyzer = 'word',
    max_features = 2000,
    smooth_idf = True,
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english',
    tokenizer = get_tokens
)

clf_pipe = pipeline.Pipeline([
    ('vect', tfidf),
    ('densify', DenseTransformer()),
    ('clf', RandomForestClassifier(n_estimators = 10, criterion='gini'))
])


current_time = datetime.datetime.now().time().isoformat()
print("Training part started      (%s)" %(current_time))
rf_model = clf_pipe.fit(data_train, y_train)
current_time = datetime.datetime.now().time().isoformat()
print("Training part finished     (%s)" %(current_time))

pred_rf = rf_model.predict(data_test)
current_time = datetime.datetime.now().time().isoformat()
print("Testin part finished       (%s)" %(current_time))

pred_rf = np.vstack((df_test.loc[:, 'id'], pred_rf)).T

print(pred_rf.shape)

np.savetxt('results/pred.randomforest.1.csv', pred_rf, fmt='%s,%1d', delimiter=',', header='id,sentiment')
