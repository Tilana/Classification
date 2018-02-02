import pandas as pd
import numpy as np
import pdb
import sys
import os
from lda.docLoader import loadConfigFile
from scripts import cnnClassification, cnnPrediction, evidenceSentencesToSummary
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from nltk.tokenize import sent_tokenize
from lda import ClassificationModel, Preprocessor, Viewer


MAX_SEQUENCE_LENGTH = 90

balance = 1
preprocessing = 1
validation = 0
splitValidationDataInSentences = 0
sentences_train_size = 75
doc_train_size = 100
random_state = 42
classificationType = 'multi'

configFile = 'dataConfig.json'

sentences_config_name = 'ICAAD_DV_sentences'
#summary_config_name = 'ICAAD_DV_summaries'

#sentences_config_name = 'ICAAD_SA_sentences'
#summary_config_name = 'ICAAD_SA_summaries'

#sentences_config_name = 'Manifesto_Minorities'


def sentenceToDocClassification():

    sentences_config = loadConfigFile(configFile, sentences_config_name)
    dataPath = sentences_config['data_path']
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')

    if classificationType == 'binary' and balance==1:
        posSample = sentences[sentences[sentences_config['TARGET']]==sentences_config['categoryOfInterest']]
        negSample = sentences[sentences[sentences_config['TARGET']] == sentences_config['negCategory']].sample(len(posSample), random_state=random_state)
        sentences = pd.concat([posSample, negSample], ignore_index=True)

    viewer = Viewer(sentences_config['DATASET'])
    viewer.printDocuments(sentences, folder='Sentences', docPath='../../' + sentences_config['DATASET'] + '/Documents')

    if preprocessing:
        preprocessor = Preprocessor()
        sentences.text = sentences.text.apply(preprocessor.cleanText)

    sentenceClassifier = ClassificationModel(target=sentences_config['TARGET'], labelOfInterest=sentences_config['categoryOfInterest'])
    sentenceClassifier.data = sentences
    sentenceClassifier.createTarget()

    sentenceClassifier.setDataConfig(sentences_config)
    sentenceClassifier.validation = validation

    sentenceClassifier.splitDataset(train_size=sentences_train_size, stratify=True)

    sentenceClassifier.max_document_length = MAX_SEQUENCE_LENGTH
    cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[2])


    pdb.set_trace()

    print 'Split Validation Data In Setences'
    data = pd.read_pickle(sentences_config['full_doc_path'])
    viewer.printDocuments(data, folder='Documents')

    validationIndices = sentenceClassifier.validationData.docID.unique()
    data = data[data.id.isin(validationIndices)]


    def splitInSentences(row):
        sentences = sent_tokenize(row.text)
        return [(row.id, row[sentences_config['label']], sentence) for sentence in sentences]

    sentenceDB = data.apply(splitInSentences, axis=1)
    sentenceDB = sum(sentenceDB.tolist(), [])
    sentenceDB = pd.DataFrame(sentenceDB, columns=['docID', sentences_config['label'], 'text'])

    sentenceDB['sentenceLength'] = sentenceDB.text.map(setSentenceLength)
    sentenceDB = sentenceDB[sentenceDB.sentenceLength.map(filterSentenceLength)]

    sentenceDB['text'] = sentenceDB['text'].str.lower()

    print 'Predict labels of sentences in validation data'
    predictedData = cnnPrediction(sentenceDB, sentences_config['label'], sentenceClassifier.output_dir)

    summaries = evidenceSentencesToSummary(predictedData, sentences_config['label'])


    summary_config = loadConfigFile(configFile, summary_config_name)

    viewer.printDocuments(summaries, folder= summary_config['ID'] + '_summaries', docPath='../../' + summary_config['DATASET'] + '/Documents')

    if preprocessing:
        preprocessor = Preprocessor()
        summaries.text = summaries.text.apply(preprocessor.cleanText)


    docClassifier = ClassificationModel(target=summary_config['TARGET'], labelOfInterest=summary_config['categoryOfInterest'])
    docClassifier.data = summaries
    docClassifier.validation = 0
    docClassifier.createTarget()

    docClassifier.setDataConfig(summary_config)
    docClassifier.splitDataset(train_size=doc_train_size, random_state=20)


    docClassifier.max_document_length = max([len(x.split(" ")) for x in docClassifier.trainData.text])
    print 'Maximal sentence length ' + str(sentenceClassifier.max_document_length)


    cnnClassification(docClassifier, BATCH_SIZE=32, ITERATIONS=3, filter_sizes=[3,4,5])

    pdb.set_trace()




if __name__=='__main__':
    sentenceToDocClassification()
