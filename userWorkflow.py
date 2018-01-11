import pandas as pd
from lda.docLoader import loadConfigFile
from predictDoc import predictDoc
from train import train

def getSentenceSample(sentences, categoryID, sentences_config):
    sentence = sentences.sample(1).iloc[0]
    isValid = sentence[sentences_config['TARGET']]==sentences_config['categoryOfInterest']
    return (sentence.text, categoryID, isValid)

def userWorkflow():

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    categoryID = 'ICAAD_DV_sentences'
    summary_config_name = 'ICAAD_DV_summaries'

    # Get Sentence dataset
    sentences_config = loadConfigFile(configFile, sentences_config_name)
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')

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


if __name__=='__main__':
    userWorkflow()
