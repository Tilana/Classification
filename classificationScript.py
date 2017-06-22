from modelSelection import modelSelection 
from preprocessing import preprocessing
from buildClassificationModel import buildClassificationModel
from FeatureExtraction import FeatureExtraction
from FeatureAnalysis import FeatureAnalysis
from validateModel import validateModel
from lda.docLoader import loadData
import pdb

targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV', 'Rape', 'DV.Restraining.Order', 'Penal.Code', 'Defilement', 'Reconciliation', 'Incest', 'Year']

whitelist = ['domestic violence', 'grievous harm', 'domestic', 'wife', 'wounding', 'bodily harm', 'batter', 'aggression', 'attack', 'protection order', 'woman']
whitelist = None

def classificationScript():

    target = targets[1]
    features = ['tfIdf']

    #dataPath = 'Documents/ICAAD/ICAAD.pkl'
    dataPath = 'Documents/RightDocs.csv'
    #modelPath = 'processedData/processedData_TF_binary'
    #modelPath = 'processedData/processedData'
    #modelPath = 'processedData/doc2vec'
    #modelPath = 'processedData/processedData_whitelist'
    #modelPath = 'processedData/SADV_whitelist'
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/RightDocs'

    data = loadData(dataPath)
    #data = data[data['Sexual.Assault.Manual'] | data['Domestic.Violence.Manual']]
    #preprocessing(data, modelPath, whitelist)
    #pdb.set_trace()
    
    data = FeatureExtraction(data[:5])
    pdb.set_trace()
    FeatureAnalysis(data)

    model  = modelSelection(modelPath, target, features, whitelist=whitelist)

    validateModel(model, features) 


if __name__=='__main__':
    classificationScript()
