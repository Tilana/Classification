import pandas as pd
import pdb
import lda
from lda.docLoader import loadConfigFile
from cnnClassification import cnnClassification
from cnnPrediction import cnnPrediction
from client import predictDoc, train
from evidenceSentencesToSummary import evidenceSentencesToSummary
from createSentenceDB import filterSentenceLength, setSentenceLength
from nltk.tokenize import sent_tokenize
from lda import ClassificationModel, Preprocessor, Viewer

def getSentenceSample(sentences, categoryID, sentences_config):
    sentence = sentences.sample(1).iloc[0]
    isValid = sentence[sentences_config['TARGET']]==sentences_config['categoryOfInterest']
    return (sentence.text, categoryID, isValid)


def userWorkflow():

    analyze = 0
    preprocessing = 1
    balanceData = 1
    validation = 1
    splitValidationDataInSentences = 0
    sentences_train_size = 100
    doc_train_size = 100

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    categoryID = 'ICAAD_DV_sentences'
    summary_config_name = 'ICAAD_DV_summaries'

    #sentences_config_name = 'ICAAD_SA_sentences'
    #summary_config_name = 'ICAAD_SA_summaries'
    #config_name = 'ICAAD_SA_sentences'
    #config_name = 'Manifesto_Minorities'

    sentences_config = loadConfigFile(configFile, sentences_config_name)
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')


    if balanceData:
        posSample = sentences[sentences[sentences_config['TARGET']]==sentences_config['categoryOfInterest']]
        negSample = sentences[sentences[sentences_config['TARGET']] == sentences_config['negCategory']].sample(len(posSample), random_state=42)
        sentences = pd.concat([posSample, negSample])


    viewer = Viewer(sentences_config['DATASET'])
    viewer.printDocuments(sentences, folder='Sentences', docPath='../../' + sentences_config['DATASET'] + '/Documents')

    # Train Classifier
    for numberSample in xrange(10):
        sentence,category,value = getSentenceSample(sentences, categoryID, sentences_config)
        train(sentence, category, value)

    # Get Full text documents
    data = pd.read_pickle(sentences_config['full_doc_path'])

    # Predict label of sentences in documents
    for numberSample in xrange(5):
        sample = data.sample(1, random_state=42).iloc[0]
        evidenceSentences = predictDoc(sample[['text', 'title']], categoryID)

    pdb.set_trace()


if __name__=='__main__':
    userWorkflow()
