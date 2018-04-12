## Classification

This module provides a framework to process, analyse, and categorize collections of documents.


## Scripts
In *dataConfig.json* the for different dataset specific parameters, like the variable to classify, the path to the document, etc. are stored.
The *scripts* folder contains different methods to analyze and classify documents. For example, with the *classificationScript.py* a sample pipeline for classification based on standard algorithms like *Logistic Regression* or *Naive Bayes* is provided. Similar in *cnnClassification.py* the pipeline for document classification based on a Convolutional Neural Network is demonstrated.

To train a convolutional neural network for sentence classification call *train.py* with a Dataframe containing the columns *sentence* for the training sentences and *label* for the corresponding categories. Furthermore the name of the model is required.

To identify relevant sentences in a document call *predictDoc.py* with the model name and a document represented as a dictionary with the field *text*. 



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


