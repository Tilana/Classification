from modelSelection import modelSelection 
from preprocessing import preprocessing
import pdb

targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV', 'Rape', 'DV.Restraining.Order', 'Penal.Code', 'Defilement', 'Reconciliation', 'Incest', 'Year'] 

def classificationScript():

    target = targets[1]
    features = ['tfIdf']

    dataPath = 'Documents/ICAAD/ICAAD.pkl'
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData_TF_binary'
    modelPath = 'processedData/processedData'
    modelPath = 'processedData/doc2vec'
    modelPath = 'processedData/SADV'

    preprocessing(dataPath, modelPath)

    classifierType, params, bestScore  = modelSelection(modelPath, target, features)
    
    pdb.set_trace()

    print True


if __name__=='__main__':
    classificationScript()
