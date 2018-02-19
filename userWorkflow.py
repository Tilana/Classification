import pandas as pd
from lda.docLoader import loadConfigFile
import tensorflow as tf
import os
import argparse as _argparse
from predictDoc import predictDoc
from train import train
from shutil import rmtree

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

    noDV = sentences[sentences.category=='Evidence.no.SADV'].sample(655)
    sentences = sentences[sentences.category=='Evidence.of.DV'].append(noDV)

    # Train Classifier
    for numberSample in xrange(10):
        sentence,category,value = getSentenceSample(sentences, categoryID, sentences_config)
        train(sentence, category, value)

    # Get Full text documents
    data = pd.read_pickle(sentences_config['full_doc_path'])
    for numberSample in xrange(3):
        sample = data.sample(1).iloc[0]
        evidenceSentences = predictDoc(sample[['text', 'title']], categoryID)

        #print evidenceSentences

    # Remove model
    rmtree(os.path.join('runs', categoryID), ignore_errors=True)
    #tf.app.flags._global_parser = _argparse.ArgumentParser()

    print 'RETRAIN'

    for numberSample in xrange(10):
        sentence,category,value = getSentenceSample(sentences, categoryID, sentences_config)
        train(sentence, category, value)
        #train(evidence['evidence']['text'], value + property, evidence['isEvidence'])



if __name__=='__main__':
    userWorkflow()
