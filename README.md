## Classification

This module provides a framework to analyse collections of documents.

The *classificationScript.py* shows a sample pipeline that includes:
 * *preprocessing.py* - normalizes text documents and computes the term relevancy (tf-idf) 
 * *FeatureExtraction.py* - extracts text properties based on named-entity recognition, regular expressions and wordlists  
 * *FeatureAnalysis.py* - analyzes which features are could indicators regarding a supervised classification task
 * *modelSelection.py* - determines best classifier and parameters for supervised classification
 * *validateModel.py* - re-trains and tests the best classifier anddisplays the accuracy measures as well as correctly and incorrectly identified documents with their extracted properties

Unsupervised classification with K-Means clustering is implemented in *clustering_ICAAD.py*. The number of clusters is variable. 
 
## Scripts
Use the following command to run the scripts:
```
python classificationScript.py
python clustering_ICAAD.py
```


## Testing
The folder *Unittests* contains the tests corresponding to each module. [*nose*](http://nose.readthedocs.org/) provides an easy way to run all tests together. <br  />

Run the tests with:
```
nosetests Unittests/
```

## Install dependencies
The code is based on different modules for machine learning and natural language processing, as well as other python libraries. To install them make sure you have [Python 2.7](https://www.python.org/download/releases/2.7/) and [pip](https://pip.pypa.io/en/stable/) installed.

Upgrade pip:
```
pip install -U
```

Install the dependencies with:
```
pip install -r requirements.txt
```

## Detailed list of dependencies
* [gensim - Topic Modeling for Humans](https://radimrehurek.com/gensim/install.html) <br />
Gensim is a free Python library designed to automatically extract semantic topics from documents by implementing Latent Semantic Analysis, Latent Dirichlet Allocation and Term-Frequency Inverse-Document Frequency models.
```
pip install --upgrade gensim
```
* [Scikit-learn - Machine Learning for Python](http://scikit-learn.org/stable/install.html) <br />
Scikit-learn is an open source machine learning library which includes various classification, regression and clustering algorithms like support vector machines, random forests, naive bayes and k-means.
```
pip install -U scikit-learn
```
* [NLTK](http://www.nltk.org/install.html) <br />
NLTK provides various tools to work with texts written in natural language. For this project tokenization, stemming and tagging are used.
```
sudo pip install -U nltk
``` 

To install NLTK Data run the Python interpreter with the commands:
```
import nltk
nltk.download()
```
* [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) <br />
pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive. To read excel files also the xlrd packages is required.
```
pip install pandas
pip install xlrd
```
* [Stanford Named Entity Recognizer (NER)](http://nlp.stanford.edu/software/CRF-NER.shtml) <br />
Stanford Named Entity Recognizer labels sequences of words in a text which represent proper names for persons, locations and organizations. The Stanford NER is included in this repository.


