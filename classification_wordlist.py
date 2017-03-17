from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def classification_wordlist():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[0]
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[0]
    alpha = 0.01 
    selectedFeatures = 'tfIdf'
    
    model = ClassificationModel(path, target)
    wordList = ['rape', 'sexual intercourse', 'carnal knowledge', 'indecent assault']
    #wordList = ['domestic violence', 'harm', 'husband', 'house', 'grevious harm']
    #model.data = model.data[model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']]
   
    print 'Preprocess Data'
    model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000, vocabulary=wordList)
    model.trainPreprocessor()
    model.data.reset_index(inplace=True)

    model.targetFeature = target
    model.createTarget()
    model.splitDataset(5000, random=False)
    nrDocs = len(model.data)
   
    print 'Train Classifier'
    model.buildClassifier(classifierType, alpha=alpha) 
    model.trainClassifier(selectedFeatures)

    print 'Evaluation'
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    #pdb.set_trace()

    model.relevantFeatures()
    #pdb.set_trace()

    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)
    pdb.set_trace()


if __name__=='__main__':
    classification_wordlist()

