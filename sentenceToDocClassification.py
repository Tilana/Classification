import pandas as pd
import numpy as np
import pdb
import sys
import os
from lda.docLoader import loadConfigFile
from scripts import cnnClassification, cnnPrediction, evidenceSentencesToSummary
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from nltk.tokenize import sent_tokenize
from lda import ClassificationModel, Preprocessor, Viewer, ImagePlotter


def sentenceToDocClassification():

    analyze = 1
    preprocessing = 1
    balanceData = 1
    validation = 0
    splitValidationDataInSentences = 0
    sentences_train_size = 20
    doc_train_size = 100
    useWord2Vec = True
    random_state = 20

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    summary_config_name = 'ICAAD_DV_summaries'

    #sentences_config_name = 'ICAAD_SA_sentences'
    #summary_config_name = 'ICAAD_SA_summaries'
    #config_name = 'ICAAD_SA_sentences'
    #config_name = 'Manifesto_Minorities'


    sentences_config = loadConfigFile(configFile, sentences_config_name)
    dataPath = sentences_config['data_path']
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')


    if balanceData:
        posSample = sentences[sentences[sentences_config['TARGET']]==sentences_config['categoryOfInterest']]
        negSample = sentences[sentences[sentences_config['TARGET']] == sentences_config['negCategory']].sample(len(posSample), random_state=random_state)
        sentences = pd.concat([posSample, negSample])


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

    sentenceClassifier.splitDataset(train_size=sentences_train_size, random_state=20)

    if analyze:
        document_lengths = [len(sentence.split(" ")) for sentence in sentences.text]
        plotter = ImagePlotter(True)
        bins = range(1,100)
        plotter.plotHistogram(document_lengths, log=False, title= sentences_config['ID'] + ' frequency of evidence sentences length', xlabel='sentence length', ylabel='frequency', bins=bins, path=None)
        print 'max: ' + str(max(document_lengths))
        print 'min 0.5: ' + str(min(document_lengths))
        print 'median: ' + str(np.median(document_lengths))
        print 'average: ' + str(np.mean(document_lengths))


    sentenceClassifier.max_document_length = max([len(x.split(" ")) for x in sentenceClassifier.trainData.text])
    print 'Maximal sentence length ' + str(sentenceClassifier.max_document_length)


    cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[2], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[2,2,2], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[1,2,3], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[2,2,3], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[2,3,4], pretrainedWordEmbeddings=useWord2Vec)

    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[3,4,5], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[4,5,6], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[5,6,7], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[6,7,8], pretrainedWordEmbeddings=useWord2Vec)
    #cnnClassification(sentenceClassifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[7,8,9], pretrainedWordEmbeddings=useWord2Vec)

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
