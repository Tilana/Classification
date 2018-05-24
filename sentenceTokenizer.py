import pandas as pd
import numpy as np
import numpy.core.numeric as _nx
from nltk.tokenize import word_tokenize
import nltk
import re

MAX_SENTENCE_LENGTH = 50
MIN_SENTENCE_LENGTH = 6


LEGAL_ABBREVATIONS = ['chap', 'distr', 'paras', 'cf', 'cfr', 'para', 'no', 'al', 'br', 'dr', 'hon', 'app', 'cr', 'crim', 'l.r', 'cri', 'cap', 'e.g', 'vol', 'd', 'a', 'ph', 'inc.v', 'prof', 'mrs', 'mrt', 'msn', 'mrj', 'msi', 'mrg', 'mra', 'mst', 'mrd', 'pp', 'seq', 'art', 'p', 'nos', 'op', 'i.e', 'tel']


def loadTokenizerWithExtraAbbrevations(language='english', abbrevations=[]):
    tokenizer = nltk.data.load('tokenizers/punkt/{0}.pickle'.format(language))
    tokenizer._params.abbrev_types.update(abbrevations)
    return tokenizer

LEGAL_TOKENIZER = loadTokenizerWithExtraAbbrevations(abbrevations=LEGAL_ABBREVATIONS)


def splitListAtPunctuationWithVarianz(wordList, sections, wordRange=5):
    wordsPerSection, extras = divmod(len(wordList), sections)
    sectionSizes = ([0] + extras * [wordsPerSection+1] + (wordsPerSection-extras) * [wordsPerSection])
    divInd = _nx.array(sectionSizes).cumsum()
    divInd = [ind for ind in divInd if ind<=len(wordList)]
    for indPos, pos in enumerate(divInd[1:len(divInd)-1]):
        wordListPart = wordList[pos-wordRange:pos+wordRange]
        indices = range(pos-wordRange, pos+wordRange+1)
        for wordPos,word in enumerate(wordListPart):
            if word == ',' or word==')' or word==']':
                divInd[indPos+1]= indices[wordPos+1]
    splittedList = np.array_split(wordList, divInd)
    return splittedList


def insertSpaceAfterPunctuation(text):
    text = re.sub(r'\.(?=[A-Z0-9])', '. ', text)
    text = re.sub(r'\)(?=[A-Z0-9])', ') ', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\(', ' (', text)
    return re.sub(r',(?=[a-zA-Z0-9])', ', ', text)

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
    listOfWords = word_tokenize(sentence)
    numberOfSplits = len(listOfWords)/MAX_SENTENCE_LENGTH + 1

    splittedSentences = splitListAtPunctuationWithVarianz(listOfWords, numberOfSplits)
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
        if len(word_tokenize(sentence)) > MAX_SENTENCE_LENGTH:
            semicolonSplit = insertNewLine(sentence, character, minimumSentenceLength)
            if len(semicolonSplit.split('\n')) > 1:
                sentences[ind] = semicolonSplit.split('\n')
    return flattenList(sentences)

def splitTooLongSentencesInChunks(sentences, MAX_SENTENCE_LENGTH):
    for ind, sentence in enumerate(sentences):
        if len(word_tokenize(sentence)) > MAX_SENTENCE_LENGTH:
            sentences[ind] = splitInChunks(sentence, MAX_SENTENCE_LENGTH)
    return flattenList(sentences)

def filterForLength(sentences, MIN_SENTENCE_LENGTH):
    return [sentence for sentence in sentences if len(sentence.split()) >= MIN_SENTENCE_LENGTH]


def tokenize(text, maxSentenceLength=50, minSentenceLength=5):
    text = insertSpaceAfterPunctuation(text)
    text = insertNewlineAfterSpeech(text)
    sentences = LEGAL_TOKENIZER.tokenize(text)
    sentences = splitAtNewline(sentences)

    sentences = splitTooLongSentencesAtCharacter(sentences, ':', minSentenceLength)
    sentences = splitTooLongSentencesAtCharacter(sentences, ';', minSentenceLength)

    sentences = splitTooLongSentencesInChunks(sentences, maxSentenceLength)

    sentences = filterForLength(sentences, MIN_SENTENCE_LENGTH)

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

