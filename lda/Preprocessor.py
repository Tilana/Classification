import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocessor:

    def __init__(self, processor='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range=(1,2), max_features=8000, token_pattern='[a-zA-Z]+', vocabulary=None):
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features, token_pattern=token_pattern, vocabulary = vocabulary, tokenizer=self.tokenize) 
        if processor=='tf':
            self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features = max_feature, token_pattern=token_pattern, tokenizer=self.tokenize, vocabulary=vocabulary)
        self.WordNet = WordNetLemmatizer()
        self.token_pattern = token_pattern


    def trainVectorizer(self, docs):
        wordCounts = self.vectorizer.fit_transform(docs)
        self.setVocabulary()
        return [docVec for docVec in wordCounts.toarray()]
                                                                              
    def vectorizeDocs(self, docs):
        wordCounts = self.vectorizer.transform(docs)
        return [docVec for docVec in wordCounts.toarray()]


    def lemmatize(self,tokens):
        return [self.WordNet.lemmatize(self.WordNet.lemmatize(self.WordNet.lemmatize(token), 'v'), 'a') for token in tokens]
                                                                                                                             
    def tokenize(self, text):
        token_pattern = re.compile(self.token_pattern)
        tokens = token_pattern.findall(text)
        #tokens = lambda doc: token_pattern.findall(text)
        return self.lemmatize(tokens)

    def setVocabulary(self):
        self.vocabulary = self.vectorizer.get_feature_names()
        
