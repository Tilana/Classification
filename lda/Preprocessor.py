import re
import cPickle as pickle
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
import string
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

numberDict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}


class Preprocessor:

    def __init__(self, processor='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range=(1,2), max_features=8000, vocabulary=None, token_pattern=r'(?u)\b\w\w+\b', binary=False):
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features, vocabulary = vocabulary, tokenizer=self.createPosLemmaTokens, binary=binary) 
        if processor=='tf':
            self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features = max_features, tokenizer=self.createPosLemmaTokens, vocabulary=vocabulary, binary=binary)
        self.WordNet = WordNetLemmatizer()
        self.token_pattern = token_pattern


    def trainVectorizer(self, docs):
        wordCounts = self.vectorizer.fit_transform(docs)
        self.setVocabulary()
        return [docVec for docVec in wordCounts.toarray()]
                                                                              

    def vectorizeDocs(self, docs):
        wordCounts = self.vectorizer.transform(docs)
        return [docVec for docVec in wordCounts.toarray()]


    def posLemmatize(self,tokens):
        lemmas = []
        for (token, tag) in tokens:
            wordnetTag = self.treebank2WordnetTag(tag)
            if wordnetTag!='':
                lemma = self.WordNet.lemmatize(token, wordnetTag)
                lemmas.append(lemma)
            else:
                lemmas.append(token)
        return lemmas 
                                                                                                                             
    def createPosLemmaTokens(self, text):
        token_pattern = re.compile(self.token_pattern)
        tokens = token_pattern.findall(text)
        posTags = self.posTagging(tokens)
        return self.posLemmatize(posTags)


    def setVocabulary(self):
        self.vocabulary = self.vectorizer.get_feature_names()


    def getVocabDict(self):
        ids = self.vectorizer.vocabulary_.values()
        words = self.vectorizer.vocabulary_.keys()
        return dict(zip(ids,words))


    def posTagging(self, tokens):
        return pos_tag(tokens)


    def wordTokenize(self, text):
        return [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]


    def treebank2WordnetTag(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''


    def save(self, path):
        self.saveVectorizer(path)
        with open(path+'_preprocessor.pkl', 'wb') as f:
            pickle.dump(self, f, -1)

    def load(self, path):
        preprocessor = pickle.load(open(path+'_preprocessor.pkl', 'rb'))
        preprocessor.vectorizer = preprocessor.loadVectorizer(path)
        preprocessor.vectorizer.tokenizer = self.createPosLemmaTokens
        return preprocessor

    def saveVectorizer(self, path):
        with open(path+'_vectorizer.pkl', 'wb') as f:
            self.vectorizer.tokenizer = []
            pickle.dump(self.vectorizer, f, -1)
        del self.vectorizer

    def loadVectorizer(self, path):
        return pickle.load(open(path+'_vectorizer.pkl', 'rb'))

    def word2num(self, word):
        return numberDict.get(word)

    def findNumbers(self, text):
        writtenNumbers = numberDict.keys()
        return set([word for word in wordpunct_tokenize(text.lower()) if word in writtenNumbers])

    
    def numbersInTextToDigits(self, text):
        wordNumbers = self.findNumbers(text)
        for word in wordNumbers:
            digitNum = str(self.word2num(word))
            text = re.sub(word, digitNum, text)
        return text

    
    def cleanText(self, text):
        tokens = self.wordTokenize(text.lower())
        posTags = self.posTagging(tokens)
        lemmas = self.posLemmatize(posTags)
        return ' '.join(lemmas)

    
    def removeHTMLtags(self, text):
        html = BeautifulSoup(text)
        return html.get_text()

    
    def removeStopwords(self, text, stopchars=None):
        if not stopchars:
            stopchars= stopwords.words('english') + list(string.punctuation) + ['--', "''", '``', "'s"]
        return [word for word in text.split() if word not in stopchars]




        
