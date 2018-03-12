import pandas as pd
from lda.docLoader import loadConfigFile
from lda import ClassificationModel, Evaluation
from predictDoc import predictDoc
from train import train
import pdb

def onlineLearning():

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    categoryID = 'ICAAD_DV_sentences'
    summary_config_name = 'ICAAD_DV_summaries'

    # Get Sentence dataset
    sentences_config = loadConfigFile(configFile, sentences_config_name)
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')

    noDV = sentences[sentences.category=='Evidence.no.SADV'].sample(655)
    sentences = sentences[sentences.category=='Evidence.of.DV'].append(noDV)

    classifier = ClassificationModel(target=sentences_config['TARGET'])
    classifier.data = sentences
    classifier.createTarget()

    classifier.splitDataset(train_size=0.97, random_state=7)

    # Train Classifier
    for numberSample in xrange(10):
        sample = classifier.trainData.sample(1)
        evidence = pd.DataFrame({'sentence':sample.text.tolist(), 'label': sample.category.tolist()})
        train(evidence, categoryID)

    # Predict label of sentences in documents
    for ind, sample in classifier.testData.iterrows():
        try:
            evidenceSentences = predictDoc(sample, categoryID)
            if len(evidenceSentences)>=1:
                classifier.testData.loc[ind, 'predLabel'] = 1
                classifier.testData.loc[ind, 'probability'] = evidenceSentences.probability.tolist()[0]
        except:
            print 'WARNING: PredictDoc.py of sample "' + str(sample.text) + '" was not successful'


    classifier.testData.predLabel.fillna(0, inplace=True)

    evaluation = Evaluation(target=classifier.testData.category.tolist(), prediction=classifier.testData.predLabel.tolist())
    evaluation.computeMeasures()
    evaluation.confusionMatrix()
    print 'Accuracy: ' + str(evaluation.accuracy)
    print 'Recall: ' + str(evaluation.recall)
    print 'Precision: ' + str(evaluation.precision)
    print evaluation.confusionMatrix



if __name__=='__main__':
    onlineLearning()
