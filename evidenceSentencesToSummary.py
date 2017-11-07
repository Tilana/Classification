import pandas as pd
import pdb

def combineSentencesToSummary(sentenceData):
    sentences = sentenceData.text.tolist()
    return ' '.join(sentences)

def getFirstElement(row):
    return row[0]

def numberOfEvidenceSentences(sentenceData):
    return sum(sentenceData.predictedLabel)


def evidenceSentencesToSummary():

    data_path = '../data/ICAAD/DV_sentencesValidationData.csv'
    file_path = '../data/ICAAD/DV_summariesValidationData.csv'
    label = 'Domestic.Violence.Manual'

    data_path = '../data/ICAAD/SA_sentencesValidationData.csv'
    file_path = '../data/ICAAD/SA_summariesValidationData.csv'
    label = 'Sexual.Assault.Manual'

    data = pd.read_csv(data_path)

    posData = data[data.predictedLabel==1]
    posDocs = posData.groupby('docId')

    numberEvidences = posDocs.apply(numberOfEvidenceSentences)
    summaries = posDocs.apply(combineSentencesToSummary)

    docIDs = posDocs.docId.unique()
    docIDs = docIDs.apply(getFirstElement)

    labels = posDocs[label].unique()
    labels = labels.apply(getFirstElement)

    dataDict = {'text': summaries, label:labels, 'nrSentences':numberEvidences}

    summaryData = pd.DataFrame(dataDict)
    summaryData.reset_index(inplace=True)
    #summaryData.to_csv(file_path)

    trueDocs = data[data[label]==1]
    trueDocIDs = trueDocs.docId.unique().tolist()

    falseNegatives = [elem for elem in trueDocIDs if elem not in docIDs]
    print 'Number of False Negatives: ' + str(len(falseNegatives))


    pdb.set_trace()



if __name__=='__main__':
    evidenceSentencesToSummary()
