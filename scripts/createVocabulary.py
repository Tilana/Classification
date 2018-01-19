import sys
sys.path.append('../')
import pandas as pd
from lda import Preprocessor
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def createVocabulary():

    PATH = '../../data/'

    UPR = pd.read_csv(PATH + 'UPR/UPR_DATABASE.csv', encoding='utf8')
    HRC = pd.read_csv(PATH + 'HRC/HRC.csv', encoding='utf8')
    ICAAD = pd.read_pickle(PATH + 'ICAAD/ICAAD.pkl')

    data = UPR.Recommendation.append(HRC.text).append(ICAAD.text)
    data.dropna(inplace=True)

    preprocessor = Preprocessor()
    cleanText = data.apply(preprocessor.cleanText)

    vectorizer = CountVectorizer(strip_accents='unicode', min_df=20)
    vectorizer.fit_transform(cleanText.tolist())

    vocabulary = vectorizer.vocabulary_
    print 'Vocabulary Length: ' + str(len(vocabulary))
    pickle.dump(vocabulary.keys(), open('../vocabulary.txt', 'wb'))


if __name__=='__main__':
    createVocabulary()
