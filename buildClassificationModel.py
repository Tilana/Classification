from lda import Viewer, ClassificationModel

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual']
    target = 'Sexual.Assault.Manual'

    modelPath = 'processedData/SA_5000'
    modelPath = 'processedData/test'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest']
    classifierType = classifierTypes[1]
    nrLabels = 5000
    nrLabels = 50

    model = ClassificationModel(path, target)
    testDoc = model.data.loc[1000]
    model.data = model.data[0:100]

    if not model.existsPreprocessedData(modelPath):
        print 'Preprocess Data'
        model.createTarget()
        model.splitDataset(nrLabels, random=False)

        model.buildPreprocessor(ngram_range=(1,2), min_df=10, max_df=0.5, max_features=8000)
        model.trainPreprocessor()
        print model.preprocessor.vocabulary
        model.preprocessTestData()
        
        model.save(modelPath)

    model2 = ClassificationModel(path,target)
    model2 = model2.load(modelPath)

    print dir(model)
    model2.preprocessTestData()
    
    
    print 'Train Classifier' 
    model.buildClassifier(classifierType, alpha=0.8)
    selectedFeatures = 'tfIdf'
    model.trainClassifier(selectedFeatures)

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

