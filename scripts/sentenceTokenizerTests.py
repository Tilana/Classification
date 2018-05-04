from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pandas as pd
import numpy as np
import re

PATH = '../../data/sampleTexts/'
data = pd.read_csv(PATH + 'sample.csv', encoding='utf8')
MAX_SENTENCE_LENGTH = 50

def insertNewlineAfterSpeech(text):
    text = re.sub(ur'\.\u2019', ur'.\u2019\n', text)
    text = re.sub(ur'\.\u201d', ur'.\u201d\n', text)
    return text

def splitAtNewline(sentences):
    sentences = [sentence.split('\n') for sentence in sentences]
    return sum(sentences, [])

def insertSpaceAfterPunctuation(text):
    return re.sub(r'\.(?=[A-Z])', '. ', text)

def splitInChunks(sentence, MAX_SENTENCE_LENGTH):
    listOfWords = sentence.split()
    splittedSentences = np.array_split(listOfWords, len(listOfWords)/MAX_SENTENCE_LENGTH + 1)
    return [' '.join(sentence.tolist()) for sentence in splittedSentences]

def insertNewLine(text, indicator, minimumSentenceLength=5):
    previousPos = 0
    shift = 0
    for match in re.finditer(indicator, text):
        pos = match.end(0)
        if len(text[previousPos:pos].split()) >= minimumSentenceLength:
            text = text[:pos + shift] + ' \n ' + text[pos + shift:]
            previousPos = pos
            shift += 3
    return text


def trainTokenizer(texts):
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(texts)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    print tokenizer._params.abbrev_types
    return tokenizer

def flattenList(listOfElems):
    flatList = []
    for elem in listOfElems:
        if isinstance(elem, (list,)):
            for item in elem:
                flatList.append(item)
        else:
            flatList.append(elem)
    return flatList


def splitTooLongSentencesAtCharacter(sentences, character):
    for ind, sentence in enumerate(sentences):
        if len(sentence.split()) > MAX_SENTENCE_LENGTH:
            semicolonSplit = insertNewLine(sentence, character)
            if len(semicolonSplit.split('\n')) > 1:
                sentences[ind] = semicolonSplit.split('\n')
    return flattenList(sentences)


for ind,doc in data.iterrows():

    print doc['_id']

    doc.text = insertSpaceAfterPunctuation(doc.text)
    doc.text = insertNewlineAfterSpeech(doc.text)
    sentences = sent_tokenize(doc.text)
    sentences = splitAtNewline(sentences)

    sentences = splitTooLongSentencesAtCharacter(sentences, ':')
    sentences = splitTooLongSentencesAtCharacter(sentences, ';')

    for ind, sentence in enumerate(sentences):
        if len(sentence.split()) > MAX_SENTENCE_LENGTH:
            print sentence
            print len(sentence.split())
            print '\n'

    sentences = [(sentence + '\n\n').encode('utf8') for sentence in sentences]

    with open(PATH + doc['collection'] + '_' + doc['_id'][:5] + '_2.txt', 'wb') as f:
        f.writelines(sentences)

    f.close()

