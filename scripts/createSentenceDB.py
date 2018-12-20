import pandas as pd
from ast import literal_eval
import os
import sys
sys.path.append('../')
import sentenceTokenizer
import pdb

EvidenceSA = 'Evidence.of.SA'
EvidenceDV = 'Evidence.of.DV'
CATEGORIES = [EvidenceSA, EvidenceDV]
TARGETS = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual']
FOLDER = ['EvidenceSA', 'EvidenceDV']


def string2list(text):
    return literal_eval(text)

def splitLabelAndSentence(evidence, docID):
    evidence = evidence.split(':')
    label = evidence[0]
    sentence = ' '.join(evidence[1:])
    return (docID, label, sentence)

def splitRow(evidenceList, docId):
    return [splitLabelAndSentence(evidence, docId) for evidence in evidenceList]

def splitList(evidenceList, docId):
    return [(evidence, docId) for evidence in evidenceList]

def flattenList(listOfLists):
    return sum(listOfLists, [])

def filterSentenceLength(length, minLength=4, maxLength=100):
    if length > minLength and length < maxLength:
        return True
    else:
        return False

def setSentenceLength(sentence):
    return len(sentence.split())

def toDataFrame(data, labels=None):
    return pd.DataFrame(data, columns=labels)

def addIDToSentences(docId, sentences):
    return [(docId, sentence) for sentence in sentences]


def createSentenceDB():

    path = '../../data/ICAAD/'
    data = pd.read_pickle(path + 'ICAAD.pkl')

    evidence = []
    nrSample = 0


    # collect evidence sentences for SA/DV cases
    for ind, category in enumerate(CATEGORIES):

        category = CATEGORIES[ind]
        target = TARGETS[ind]
        foldername = FOLDER[ind]

        subData = data.dropna(subset=[category])

        subData[category] = subData[category].str.lower()
        subData[category] = subData[category].apply(string2list)
        subData['sentences'] = subData.apply(lambda x: splitRow(x[category], x.id), axis=1)
        sentences = flattenList(subData['sentences'].tolist())
        currDF = toDataFrame(sentences, ['id', 'label', 'sentence'])
        currDF['category'] = category

        currDF['sentenceLength'] = currDF.sentence.map(setSentenceLength)
        currDF = currDF[currDF.sentenceLength.map(filterSentenceLength)]

        groups = currDF.groupby('label')
        for group in groups:
            print group[0] + ': ' + str(len(group[1]['id'].unique()))

        nrSample = nrSample + len(currDF)
        evidence.append(currDF)


    # Randomly select non SA/DV sentences from cases
    category = 'Non.SA.DV.case'
    nonSADV = data[data[category]]
    nonSADV['sentences'] = nonSADV.text.apply(sentenceTokenizer.tokenize, maxSentenceLength=40, minSentenceLength=5)
    nonSADV['sentences'] = nonSADV.apply(lambda x: addIDToSentences(x.id, x.sentences), axis=1)

    sentences = flattenList(nonSADV.sentences.tolist())
    currDF = toDataFrame(sentences, ['id', 'sentence'])
    currDF['category'] = 'Evidence.no.SADV'
    currDF['label'] = 'Evidence.no.SADV'

    currDF['sentenceLength'] = currDF.sentence.map(setSentenceLength)
    currDF = currDF[currDF.sentenceLength.map(filterSentenceLength)]

    currDF['sentence'] = currDF['sentence'].str.lower()
    evidence.append(currDF.sample(nrSample))

    # Create final database with all categories
    sentenceDB= pd.concat(evidence)
    sentenceDB['sentence'] = sentenceDB.sentence.str.strip()
    sentenceDB.to_csv(path + 'ICAAD_labeledSentences.csv', index=False)


if __name__=='__main__':
    createSentenceDB()
