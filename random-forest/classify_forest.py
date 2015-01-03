
import os
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Read the labeled training data
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, 
                    delimiter="\t", quoting=3)


print(train.shape)


