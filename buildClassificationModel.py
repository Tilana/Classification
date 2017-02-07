from lda import Viewer, ClassificationModel

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = 'Sexual.Assault.Manual'
    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest']
    classifierType = classifierTypes[1]
    nrLabels = 5000

    model = ClassificationModel(path, target)
    model.createTarget()
    model.splitDataset(nrLabels, random=False)

    model.buildPreprocessor(ngram_range=(1,2), min_df=10, max_df=0.5, max_features=8000)
    model.trainPreprocessor()
    print model.preprocessor.vocabulary
    print model.preprocessor.vectorizer.get_stop_words()
    
    model.buildClassifier(classifierType, alpha=0.2)
    selectedFeatures = 'tfIdf'
    model.trainClassifier(selectedFeatures)

    model.preprocessTestData()
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    viewer = Viewer(classifierType)
   
    features = ['Court', 'Year', 'Sexual.Assault.Manual', 'predictedLabel', 'tag']
    viewer.printDocuments(model.testData,features)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

