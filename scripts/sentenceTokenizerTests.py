from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pandas as pd
import re

PATH = '../../data/sampleTexts/'
data = pd.read_csv(PATH + 'sample.csv', encoding='utf8')
MAX_SENTENCE_LENGTH = 30

def insertNewlineAfterSpeech(text):
    text = re.sub(ur'\.\u2019', ur'.\u2019\n', text)
    text = re.sub(ur'\.\u201d', ur'.\u201d\n', text)
    return text

def splitAtNewline(sentences):
    sentences = [sentence.split('\n') for sentence in sentences]
    return sum(sentences, [])

def insertSpaceAfterPunctuation(text):
    return re.sub(r'\.(?=[A-Z])', '. ', text)

def splitTooLongSentencesAtSemicolon(sentences, MAX_SENTENCE_LENGTH):
    sentences = [re.sub(';', ';\n', sentence) for sentence in sentences if len(sentence)>MAX_SENTENCE_LENGTH]
    return splitAtNewline(sentences)

def splitTooLongSentencesAtColon(sentences, MAX_SENTENCE_LENGTH):
    sentences = [re.sub(':', ':\n', sentence) for sentence in sentences if len(sentence)>MAX_SENTENCE_LENGTH]
    return sentences

def trainTokenizer(texts):
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(texts)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    print tokenizer._params.abbrev_types
    return tokenizer


for ind,doc in data.iterrows():

    doc.text = insertSpaceAfterPunctuation(doc.text)
    doc.text = insertNewlineAfterSpeech(doc.text)
    sentences = sent_tokenize(doc.text)
    sentences = splitAtNewline(sentences)

    sentences = splitTooLongSentencesAtSemicolon(sentences, MAX_SENTENCE_LENGTH)
    #sentences = splitTooLongSentencesAtColon(sentences, MAX_SENTENCE_LENGTH)

    sentences = [(sentence + '\n\n').encode('utf8') for sentence in sentences]

    with open(PATH + doc['collection'] + '_' + doc['_id'][:5] + '3.txt', 'wb') as f:
        f.writelines(sentences)

    f.close()


