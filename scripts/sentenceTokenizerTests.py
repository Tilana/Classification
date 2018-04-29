from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pandas as pd
import re
import pdb

PATH = '../../data/sampleTexts/'
data = pd.read_csv(PATH + 'sample.csv', encoding='utf8')

def splitAfterSpeech(listOfSentences):
    sentences = [sentence.split(ur"""\.[\u201d\"'\u2019]""") for sentence in listOfSentences]
    sentences = [sentence.split(ur'and') for sentence in listOfSentences]
    return sum(sentences, [])

def insertSpaceAfterPunctuation(text):
    return re.sub(r'\.(?=[A-Z])', '. ', text)

def insertSpaceAfterSpeech(text):
    return re.sub(ur"\.\u201d", ur'.\u201d ', text)

def trainTokenizer(texts)
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(texts)

    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    print tokenizer._params.abbrev_types

    return tokenizer


for ind,doc in data.iterrows():

    #doc.text = insertSpaceAfterPunctuation(doc.text)
    sentences = sent_tokenize(doc.text)
    sentences = splitAfterSpeech(sentences)
    sentences = [(sentence + '\n\n').encode('utf8') for sentence in sentences]

    with open(PATH + doc['collection'] + '_' + doc['_id'][:5] + '3.txt', 'wb') as f:
        f.writelines(sentences)

    f.close()

pdb.set_trace()



