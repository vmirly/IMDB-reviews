
import pandas as pd
from bs4 import BeautifulSoup

# Read the labeled training data
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)


print(train.shape)


