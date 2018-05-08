import pandas as pd
import numpy as np
import nltk
import re

MAX_SENTENCE_LENGTH = 50
MIN_SENTENCE_LENGTH = 6

LEGAL_ABBREVATIONS = ['chap', 'distr', 'paras', 'cf', 'cfr', 'para', 'no', 'al', 'br', 'dr', 'hon', 'app', 'cr', 'crim', 'l.r', 'cri', 'cap', 'e.g', 'vol', 'd', 'a', 'ph']

def loadTokenizerWithExtraAbbrevations(language='english', abbrevations=[]):
    tokenizer = nltk.data.load('tokenizers/punkt/{0}.pickle'.format(language))
    tokenizer._params.abbrev_types.update(abbrevations)
    return tokenizer

LEGAL_TOKENIZER = loadTokenizerWithExtraAbbrevations(abbrevations=LEGAL_ABBREVATIONS)

def insertSpaceAfterPunctuation(text):
    return re.sub(r'\.(?=[A-Z])', '. ', text)

def insertNewlineAfterSpeech(text):
    text = re.sub(ur'\.\u2019', ur'.\u2019\n', text)
    text = re.sub(ur'\.\u201d', ur'.\u201d\n', text)
    return text

def insertNewLine(text, indicator, minimumSentenceLength):
    previousPos = 0
    shift = 0
    for match in re.finditer(indicator, text):
        pos = match.end(0)
        if len(text[previousPos:pos].split()) >= minimumSentenceLength:
            text = text[:pos + shift] + ' \n ' + text[pos + shift:]
            previousPos = pos
            shift += 3
    return text

def splitAtNewline(sentences):
    sentences = [sentence.split('\n') for sentence in sentences]
    return sum(sentences, [])

def splitInChunks(sentence, MAX_SENTENCE_LENGTH):
    listOfWords = sentence.split()
    splittedSentences = np.array_split(listOfWords, len(listOfWords)/MAX_SENTENCE_LENGTH + 1)
    return [' '.join(sentence.tolist()) for sentence in splittedSentences]

def flattenList(listOfElems):
    flatList = []
    for elem in listOfElems:
        if isinstance(elem, (list,)):
            for item in elem:
                flatList.append(item)
        else:
            flatList.append(elem)
    return flatList

def splitTooLongSentencesAtCharacter(sentences, character, minimumSentenceLength=5):
    for ind, sentence in enumerate(sentences):
        if len(sentence.split()) > MAX_SENTENCE_LENGTH:
            semicolonSplit = insertNewLine(sentence, character, minimumSentenceLength)
            if len(semicolonSplit.split('\n')) > 1:
                sentences[ind] = semicolonSplit.split('\n')
    return flattenList(sentences)

def splitTooLongSentencesInChunks(sentences, MAX_SENTENCE_LENGTH):
    for ind, sentence in enumerate(sentences):
        if len(sentence.split()) > MAX_SENTENCE_LENGTH:
            sentences[ind] = splitInChunks(sentence, MAX_SENTENCE_LENGTH)
    return flattenList(sentences)


def tokenize(text, maxSentenceLength=50, minSentenceLength=5):
    text = insertSpaceAfterPunctuation(text)
    text = insertNewlineAfterSpeech(text)
    sentences = LEGAL_TOKENIZER.tokenize(text)
    sentences = splitAtNewline(sentences)

    sentences = splitTooLongSentencesAtCharacter(sentences, ':', minSentenceLength)
    sentences = splitTooLongSentencesAtCharacter(sentences, ';', minSentenceLength)
    sentences = splitTooLongSentencesInChunks(sentences, maxSentenceLength)

    return sentences


def sampleScript():

    PATH = '../data/sampleTexts/'
    data = pd.read_csv(PATH + 'sample.csv', encoding='utf8')

    for ind,doc in data.iterrows():

        sentences = tokenize(doc.text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
        sentences = [(sentence + '\n\n').encode('utf8') for sentence in sentences]

        with open(PATH + doc['collection'] + '_' + doc['_id'][:5] + '_2.txt', 'wb') as f:
            f.writelines(sentences)

        f.close()


if __name__=='__main__':
    sampleScript()

