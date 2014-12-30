
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


# #### Unit test for tokenizer:

# In[9]:

get_tokens("What's in a name? That which we call a rose by any other name would smell as sweet.")

## Note: you need to download punkt package in nltk:
# import nltk
# nltk.download(punkt)


# ### 2.2 TF-IDF Feature Extraction


# #### Unit test for TF-IDF:


# ### 2.3 Evaluate TF-IDF on the reviews


# ## 3. Hyper-parameter Optimization using KFold Cross Validation

# In[12]:

from sklearn import pipeline
from sklearn import metrics
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB

import datetime
import gc # python's garbage collector




# ## 4. Extensive Grid Search



tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
    encoding = 'utf-8',
    decode_error = 'replace',
    strip_accents = 'ascii',
    analyzer = 'word',
    smooth_idf = True,
    tokenizer = get_tokens
)

param_grid = {
    'vect__max_df':[0.2, 0.4, 0.5, 0.6, 0.8],
    'vect__sublinear_tf':[True, False],
    'vect__max_features':[5000, 10000, 20000, 40000, None],
    'vect__ngram_range':[(1,1), (1,2)],
    'clf__alpha':[0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
}


clf_pipe = pipeline.Pipeline([
        ('vect', tfidf),
        ('clf', MultinomialNB())
])

print(clf_pipe.get_params())

auc_scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True)

grs = grid_search.GridSearchCV(clf_pipe, 
                  param_grid, 
                  n_jobs=-1, 
                  verbose=1,
                  scoring=auc_scorer,
                  cv=5
)


gc.collect()



grs.fit(data_train, y_train)



print("Best score: %0.3f" % grs.best_score_)
print("Best parameters set:")
best_param = grs.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_param[param_name]))

