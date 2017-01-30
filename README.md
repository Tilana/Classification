#lda

lda provides a framework to analyse collections of documents:
 * *topicModeling.py* - uses gensim to extract the most relevant topics
 * *frequencyAnalysis.py* - returns most frequent words based on the Stanford Named-Entity Recognizer
 * *classification.py* - analysis and classification of document features with scikit-learn
 
##Dependencies

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


## Scripts
Use the following command to run the scripts:
```
python topicModeling.py
python frequencyAnalysis.py
python classification.py
```
In the **TopicModeling** and **frequencyAnalysis** files the following parameters can be adapted and are stored as an *info* object:
* *data* - specifies the name of the collection. At different collections are available: *ICAAD*, *NIPS*, *scifibooks*
* *preprocess* - flag for preprocessing: * 0 - loads preprocessed documents if found * 1 - runs preprocessing and saves documents
* *startDoc* - index of document to start upload
* *numberDoc* - number of documents for the preprocessing. Default is *None* to load all documents
* *specialChars* - remove these characters from text
* *includeEntities* - when set to 1 the Stanford named-entity recognizer extracts the names, organizations and locations from the documents
* *lowerfilter* - removes all words from dictionary which appear in less than *n* (int) documents
* *uperfilter* - removes all words from dictionary which appear in more than *x* (float) per cent of the documents
* *modelType* - *LDA* for Latent Dirichlet Allocation or *LSI* for Latent Semantic Indexing
* *numberTopics* - specify how many topics are extracted
* *tfidf* - use term-frequency inverse-document frequency weighting to train the model
* *passes* - indicates how often the algorithm is trained
* *iterations* - maximal number of iterations in each step of the LDA, less iterations are done when the parameter rho is exceeded
* *online* - splits data into chunks for faster convergence
* *chunksize* - size of chunks for online training
* *multicore* - use multi core processing to speed up training


* *whiteList* - use only the words in the white list to build the dictionary 
* *analyseDictionary* - displays document frequency of words
* *categories* - list of category words to describe the topics


The **classification** script by default loads a csv file.
* *path* - specifies the location and name of the file
* *predictColumn* - determines which column is selected to be classified
* *dropList* - contains all columns that are ignored in the classification


## Testing
The folder *Unittests* contains the tests corresponding to each module. [*nose*](http://nose.readthedocs.org/) provides an easy way to run all tests together. <br  />
Install *nose* with:
```
pip install nose
```
Run the tests with:
```
nosetests Unittests/
```
