from modelSelection import modelSelection 
from preprocessing import preprocessing
from buildClassificationModel import buildClassificationModel
from FeatureExtraction import FeatureExtraction
from FeatureAnalysis import FeatureAnalysis
from validateModel import validateModel
import pandas as pd
import pdb

targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV', 'Rape', 'DV.Restraining.Order', 'Penal.Code', 'Defilement', 'Reconciliation', 'Incest', 'Year']

whitelist = ['domestic violence', 'grievous harm', 'domestic', 'wife', 'wounding', 'bodily harm', 'batter', 'aggression', 'attack', 'protection order', 'woman']
whitelist = None

def classificationScript():

    target = targets[1]
    features = ['tfIdf']

    dataPath = 'Documents/ICAAD/ICAAD.pkl'
    modelPath = 'processedData/processedData_TF_binary'
    modelPath = 'processedData/processedData'
    modelPath = 'processedData/doc2vec'
    modelPath = 'processedData/SADV'

    #data = pd.read_pickle(dataPath)
    #preprocessing(data, modelPath)
    
    #data = FeatureExtraction(data[:10])

    #FeatureAnalysis(data)

    model  = modelSelection(modelPath, target, features, whitelist=whitelist)

    #pdb.set_trace()
    
    validateModel(model, features) 

    #pdb.set_trace()


if __name__=='__main__':
    classificationScript()
