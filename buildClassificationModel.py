#from lda import ClassificationModel, Viewer, Info
from lda import Viewer, ClassificationModel

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = 'Sexual.Assault.Manual'
    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest']
    classifierType = classifierTypes[1]
    nrLabels = 5000

    model = ClassificationModel(path, target, [])
    model.createTarget()
    model.splitDataset(nrLabels, random=False)

    model.buildVectorizer(ngram_range=(1,1), min_df=20)
    model.trainVectorizer()

    model.buildClassifier(classifierType) #, alpha)
    selectedFeatures = 'tfIdf'
    model.trainClassifier(selectedFeatures)

    model.vectorizeDocs()
    model.predict(selectedFeatures)
    model.evaluate()

    viewer = Viewer(classifierType)
   
    features = ['Court', 'Year', 'Sexual.Assault.Manual', 'predictedLabel', 'probability']
    viewer.printDocuments(testData,features)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

