
import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class Word2VecConverter(object):
    """ 
    Class for text processing
    """
    @staticmethod

    def convert_to_wordlist( review, remove_stopwords=False ):
	text = BeautifulSoup(review).get_text()
	text = re.sub("[^a-zA-Z]"," ", text)
        wordlist = review_text.lower().split()
 
	# Remove stop words 
	if remove_stopwords:
	    stops = set(stopwords.words("english"))
	    wordlist = [w for w in wordlist if not w in stops]

	return(wordlist)
