## Classification

This module provides a framework to process, analyse, and categorize collections of documents.




## Main Scripts
**semantic-search.py**  - computes semantic similarity of a search term and sentences in a database.

**compareDataframes.py** - shows the classification differences (shifts from FN to TP, etc.) of two semantic search models.

**explore_fasttext.py**  - change the preprocessing steps (removing stopwords, stemming, splitting sentence in half, etc.) of specified sample sentences to see their effect on semantic similarity

**explore_corpus.py** - get an overview about the number of (unique) words in a document collection. Also, get frequency of specific terms and randomly select context phrases.

**explore_we_model.py** - visualize vector representations of words with heatmaps, show the influence of averaging and a principal component analysis




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
pip install --user -r requirements.txt
```

### Installing NLTK data
Install *nltk* and to download the required modules, open python and type the following commands:
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

```

### Installing FLASK framework
Install flask and run the application
```
pip install Flask
FLASK_APP=routes.py flask run --port 4000

```

### Download pre-trained Word2Vec model
The word embeddings used for sentence classification with a convolutional neural network can either be trained on the specific collection or implemented by using a pre-trained model.
Google provides such pre-trained word embeddings which are trained on parts of the Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. The archive (1.5 GB) is [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) available.
To make use of it, just download the file and unpack it in the **Word2Vec** folder in the main directory.

FastText also provides pretrained word embeddings for different languages which can be found [here](https://fasttext.cc/docs/en/crawl-vectors.html). As fasttext is trained on character n-grams it is possible to provide vectors of words that were not included in the training data. To load this model in an efficient way a daemon process build on *Pyro* is used.
Start the daemon process with:
```
python lda/WordEmbedding.py
```
Then run the respective script, e.g.:
```
python onlineLearning.py
```

## Detailed list of dependencies
* [Tensorflow - An Open-Source Software Library for Machine Intelligence](https://www.tensorflow.org/install/) <br />
  TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks.
```
pip install tensorflow      # Python 2.7; CPU support
pip install tensorflow-gpu  # Python 2.7; GPU support
```

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

* [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) <br />
  pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive. To read excel files also the xlrd packages is required.
```
pip install pandas
pip install xlrd
```
* [Stanford Named Entity Recognizer (NER)](http://nlp.stanford.edu/software/CRF-NER.shtml) <br />
  Stanford Named Entity Recognizer labels sequences of words in a text which represent proper names for persons, locations and organizations. The Stanford NER is included in this repository.


