import pandas as pd
import pdb
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

    pdb.set_trace()

    #sentenceClassifier = ClassificationModel(target=sentences_config['TARGET'], labelOfInterest=sentences_config['categoryOfInterest'])
    #sentenceClassifier.data = sentences
    #sentenceClassifier.createTarget()

    #sentenceClassifier.setDataConfig(sentences_config)
    #sentenceClassifier.validation = validation

    #sentenceClassifier.splitDataset(train_size=sentences_train_size, random_state=20)


    #if analyze:
    #    coi = sentenceClassifier.sentences[sentenceClassifier.sentences[sentences_config['TARGET']]==sentences_config['categoryOfInterest']]
    #    document_lengths = [len(word_tokenize(sentence)) for sentence in coi.text]
    #    plotter = ImagePlotter(True)
    #    #figure_path = path=os.path.join(PATH, sentences_config['DATASET'], 'figures', data_config['ID'] + '_evidenceSentences' + '.png')
    #    bins = range(1,100)
    #    plotter.plotHistogram(document_lengths, log=False, title= sentences_config['ID'] + ' frequency of evidence sentences length', xlabel='sentence length', ylabel='frequency', bins=bins, path=None)
    #    print 'max: ' + str(max(document_lengths))
    #    print 'min 0.5: ' + str(min(document_lengths))
    #    print 'median: ' + str(np.median(document_lengths))

    #sentenceClassifier.max_document_length = max([len(x.split(" ")) for x in sentenceClassifier.trainData.text])
    #print 'Maximal sentence length ' + str(sentenceClassifier.max_document_length)


    cnnClassification(sentenceClassifier, ITERATIONS=3, BATCH_SIZE=64, filter_sizes=[2,3,4,5])


    print 'Split Validation Data In Sentences'
    data = pd.read_pickle(sentences_config['full_doc_path'])
    validationIndices = sentenceClassifier.validationData.docID.unique()
    data = data[data.id.isin(validationIndices)]

    doc = data.sample(1, random_state=42).iloc[0]
    evidenceSentences = predictDoc(doc, categoryID)

    pdb.set_trace()




if __name__=='__main__':
    userWorkflow()
