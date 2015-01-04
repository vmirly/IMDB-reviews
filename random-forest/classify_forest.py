
import os
import sys
import pandas as pd

import word2vec as wv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import pandas as pd
import numpy as np


### setting the parameter dictionary
param_dict = {
    'use_idf': bool(sys.argv[1]),
    'ngram_range': (int(sys.argv[5]), int(sys.argv[6])),
    'max_features': int(sys.argv[7]),
    'max_df': float(sys.argv[8]),
    'min_df': int(sys.argv[9])
}




# Read the labeled training data
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, 
                    delimiter="\t", quoting=3)


print(train.shape)


test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]

tfv = TfidfVectorizer(
        min_df=param_dict['min_df'],
        max_df=param_dict['max_df'],
        max_features=param_dict['max_features'],
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=param_dict['ngram_range'],
        use_idf=param_dict['use_idf'],
        smooth_idf=True,
        sublinear_tf=True,
        stop_words = 'english'
)


X_all = traindata + testdata
lentrain = len(traindata)
