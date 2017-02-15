from lda import Viewer, ClassificationModel
import pdb

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual']
    target = 'Sexual.Assault.Manual'
    target = 'Domestic.Violence.Manual'

    modelPath = 'processedData/SA_5000'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[2]
    alpha = 0.4 
    selectedFeatures = 'tfIdf'
    nrLabels = 5000
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'predictedLabel', 'tag']

    model = ClassificationModel(path, target)
    #model.data = model.data[0:100]

    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=10, max_df=0.5, max_features=8000)
        model.trainPreprocessor()
        print model.preprocessor.vocabulary
        
        model.save(modelPath)

    model = model.load(modelPath)
    
    model.createTarget()
    model.splitDataset(nrLabels, random=False)
    
    print 'Train Classifier' 
    model.buildClassifier(classifierType, alpha=alpha)
    model.trainClassifier(selectedFeatures)

    print 'Predict Labels'
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    print 'Display Results'
    viewer = Viewer(classifierType)
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

