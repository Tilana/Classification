import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocessor:

    def __init__(self, processor='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range=(1,2), max_features=8000, vocabulary=None, token_pattern=r'(?u)\b\w\w+\b'):
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features, vocabulary = vocabulary, tokenizer=self.createPosLemmaTokens) 
        if processor=='tf':
            self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features = max_feature, tokenizer=self.createPosLemmaTokens, vocabulary=vocabulary)
        self.WordNet = WordNetLemmatizer()
        self.token_pattern = token_pattern


    def trainVectorizer(self, docs):
        wordCounts = self.vectorizer.fit_transform(docs)
        self.setVocabulary()
        return [docVec for docVec in wordCounts.toarray()]
                                                                              
    def vectorizeDocs(self, docs):
        wordCounts = self.vectorizer.transform(docs)
        return [docVec for docVec in wordCounts.toarray()]


    # def lemmatize(self,tokens):
    #     return [self.WordNet.lemmatize(self.WordNet.lemmatize(self.WordNet.lemmatize(token), 'v'), 'a') for token in tokens]


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
        
