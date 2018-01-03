import textrank
import pandas as pd
from nltk import sent_tokenize

def printSummaries(collection):
    for ind, text in enumerate(collection):
        print 'Sample {:d}'.format(ind)
        print textrank.extract_sentences(text, summary_length=200)
        #print textrank.extract_key_phrases(text)
        print ''


def getSummaries(collection):
    return [(row.id, textrank.extract_sentences(row.text, summary_length=200)) for row in collection.itertuples()]


def splitInSentences(collection):
    sentences = []
    for ind, text in collection:
        for sentence in sent_tokenize(text):
            sentences.append((ind, sentence))
    return sentences


def getSampleTexts(data, nrSamples=500):
    sample = data.sample(nrSamples)
    return sample.text.tolist()


def summarizeTexts():

    path = '../../data/ICAAD/'
    dataPath = path + 'ICAAD.pkl'
    data = pd.read_pickle(dataPath)


    print 'No SA/DV Documents'
    noSA = data[data['Sexual.Assault.Manual']==False]
    noSADV = noSA[noSA['Domestic.Violence.Manual']==False]
    noSADV = noSADV.sample(1000)
    summaries = getSummaries(noSADV)
    noSADV_sentences = splitInSentences(summaries)

    noSADV_data = pd.DataFrame(noSADV_sentences, columns=['id', 'sentence'])
    noSADV_data.to_csv(dataPath + 'noSADV_summaries.csv')


    print 'SA Documents'
    SA = data[data['Sexual.Assault.Manual']]
    SA_summaries = getSummaries(SA)
    SA_sentences = splitInSentences(SA_summaries)

    SA_data = pd.DataFrame(SA_sentences, columns=['id', 'sentence'])
    SA_data.to_csv(dataPath + 'SA_summaries.csv')


    print 'DV Documents'
    DV = data[data['Domestic.Violence.Manual']]
    DV_summaries = getSummaries(DV)
    DV_sentences = splitInSentences(DV_summaries)

    DV_data = pd.DataFrame(DV_sentences, columns=['id', 'sentence'])
    DV_data.to_csv(dataPath + 'DV_summaries.csv')


if __name__=='__main__':
    summarizeTexts()
