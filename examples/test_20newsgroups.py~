import logging
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn import metrics



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

categories = None 

remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

categories = data_train.target_names

print(categories)

y_train, y_test = data_train.target, data_test.target



### Vectorizing:
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)

print("n_samples: %d, n_features: %d" % X_train.shape)


### Transforming the test dataset:
X_test = vectorizer.transform(data_test.data)

print("n_test: %d, n_features: %d" %X_test.shape)



### Train the classifier and test:
def do_classify(clf):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    score = metrics.f1_score(y_test, pred)

    return(score)




########## Running Classifiers ###########

s1 = do_classify(BernoulliNB(alpha=.01))
print("\n\tBernoulliNB:\t F1-score:   %0.3f" % s1)

s2 = do_classify(MultinomialNB(alpha=.01))
print("\n\tMultinomialNB:\t F1-score:   %0.3f" % s2)

s3 = do_classify(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
print("\n\tSGD\t F1-score:   %0.3f" % s3)

s4 = do_classify(KNeighborsClassifier(n_neighbors=10))
print("\n\tKNN\t F1-score:   %0.3f" % s4)
