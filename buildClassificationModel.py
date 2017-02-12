from lda import Viewer, ClassificationModel

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual']
    target = 'Sexual.Assault.Manual'

    modelPath = 'processedData/SA_5000'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest']
    classifierType = classifierTypes[1]
    nrLabels = 5000

    model = ClassificationModel(path, target)

    if not model.existsPreprocessedData(modelPath):
        print 'Preprocess Data'
        model.createTarget()
        model.splitDataset(nrLabels, random=False)

        model.buildPreprocessor(ngram_range=(1,2), min_df=10, max_df=0.5, max_features=8000)
        model.trainPreprocessor()
        print model.preprocessor.vocabulary
        print model.preprocessor.vectorizer.get_stop_words()

        print dir(model)
        model.saveTrainTestData(modelPath)

    else:
        print 'Load Preprocessed Data'
        model.loadTrainTestData(modelPath)
    
    print 'Train Classifier' 
    model.buildClassifier(classifierType, alpha=0.2)
    selectedFeatures = 'tfIdf'
    model.trainClassifier(selectedFeatures)

    print 'Preprocess Test Data'
    model.preprocessTestData()
    model.saveTrainTestData(modelPath)

    print 'Predict Labels'
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    viewer = Viewer(classifierType)
   
    features = ['Court', 'Year', 'Sexual.Assault.Manual', 'predictedLabel', 'tag']
    viewer.printDocuments(model.testData,features)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

