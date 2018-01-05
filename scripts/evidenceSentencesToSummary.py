import pandas as pd
import pdb

def combineSentencesToSummary(sentenceData):
    sentences = sentenceData.text.tolist()
    return ' '.join(sentences)

def getFirstElement(row):
    return row[0]

def numberOfEvidenceSentences(sentenceData):
    return sum(sentenceData.predictedLabel)


def evidenceSentencesToSummary(data, label):

    posData = data[data.predictedLabel==1]
    posDocs = posData.groupby('docID')

    numberEvidences = posDocs.apply(numberOfEvidenceSentences)
    summaries = posDocs.apply(combineSentencesToSummary)

    docIDs = posDocs.docID.unique()
    docIDs = docIDs.apply(getFirstElement)

    labels = posDocs[label].unique()
    labels = labels.apply(getFirstElement)

    dataDict = {'text': summaries, label:labels, 'nrSentences':numberEvidences}

    summaryData = pd.DataFrame(dataDict)
    summaryData.reset_index(inplace=True)
    summaryData['id'] = summaryData.index

    trueDocs = data[data[label]==1]
    trueDocIDs = trueDocs.docID.unique().tolist()

    falseNegatives = [elem for elem in trueDocIDs if elem not in docIDs]
    print 'Number of False Negatives: ' + str(len(falseNegatives))

    return summaryData



