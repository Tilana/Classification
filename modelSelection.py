from lda import Viewer, ClassificationModel, FeatureExtractor
from lda.dataframeUtils import toCSV
import pandas as pd
import pdb

def modelSelection():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV', 'Rape', 'DV.Restraining.Order', 'Penal.Code', 'Defilement', 'Reconciliation', 'Incest', 'Year']
    target = targets[11]
    #modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData_TF_binary'
    modelPath = 'processedData/processedData'
    #modelPath = 'processedData/doc2vec'
    pca=False
    pcaComponents = 200 
    resultPath = 'modelSelection/AllDocs_Year_tfIdf.csv'
    if pca:
        resultPath = 'modelSelection/SADV_Age_PCA' + str(pcaComponents)+ '.csv'
    #resultPath = 'modelSelection/' + target + '_SADV_3cv_400docs.csv'

    classifierTypes = ['LogisticRegression', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'DecisionTree', 'SVM', 'kNN']
    classifierTypes = ['LogisticRegression', 'BernoulliNB', 'RandomForest', 'DecisionTree', 'SVM', 'kNN']
    selectedFeatures = 'tfIdf'
    #selectedFeatures = ['docVec', 'tfIdf']
    nrTrainingDocs = 6000 
    
    model = ClassificationModel(path, target)
    results = pd.DataFrame(columns=classifierTypes, index=['Best score', 'params', 'Test Accuracy', 'Test Precision', 'Test Recall'])
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(vecType='tf', ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000, binary=True)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)


    model = model.load(modelPath)
    #model.data = model.data[model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']]
    model.targetFeature = target
    model.createTarget()

    #pdb.set_trace()
    
    model.splitDataset(nrTrainingDocs, random=True)
    nrDocs = len(model.data)

    for classifierType in classifierTypes:
        print classifierType

        #pdb.set_trace()

        model.buildClassifier(classifierType) 
        weightedFscore = model.weightFScore(2)
        (bestScore, params) = model.gridSearch(selectedFeatures, scoring=weightedFscore, scaling=False, pca=pca, components=pcaComponents)
        print('Best score: %0.3f' % bestScore)
        print params

        model.predict(selectedFeatures)
        model.evaluate()
        print 'Accuraccy: {:f}'.format(model.evaluation.accuracy)
        print 'Precision: {:f}'.format(model.evaluation.precision)
        print 'Recall: {:f}'.format(model.evaluation.recall)

        results[classifierType] = [bestScore, params, model.evaluation.accuracy, model.evaluation.precision, model.evaluation.recall]

        pdb.set_trace()

     
    print results
    toCSV(results,resultPath)

    pdb.set_trace()

    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
    viewer.printDocuments(model.testData, displayFeatures)
    model.evaluation.confusionMatrix()
    viewer.classificationResults(model)

    pdb.set_trace()


if __name__=='__main__':
    modelSelection()

