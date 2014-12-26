IMDB-reviews
============

Sentiment analysis on IMDB movie reviews


# Outline

### Dataset

### Feature Extraction

  * Count Vectorizer
  * TF-IDF

### Classification Models

  * naive Bayes 
	* Multivariate Bernoullii Distribution
	* Multonmoal Distribution
  * Random Forest
  * Deep Learning

### Hyperparameter Optimzation

  * Additive Smoothing Parameter
  * Threshold

### Conclusion


# Theory

## Naive Bayes Classification

$x_i$: Feature vector for datum $i$  
$x_i \in \omega_j | $


$$P(\omega_j | x_i) = \frac{P(x_i | \omega_j) . P(\omega_j)}{P(x_i)}$$

$P(x_i)$ plays as a normalization factor, and therefore is the same for all 
