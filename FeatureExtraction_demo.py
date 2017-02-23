from lda import Viewer, FeatureExtractor, Preprocessor
from lda import namedEntityRecognition as ner
import pandas as pd
import pdb

def FeatureExtraction_demo():

    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)
    indexSA = data[data['Sexual.Assault.Manual']].index.tolist()
    
    ind = indexSA[35] 
    doc = data.loc[ind]

    preprocessor = Preprocessor()
    cleanText = preprocessor.cleanText(doc.text)
    cleanText = preprocessor.numbersInTextToDigits(cleanText)
   
    extractor = FeatureExtractor()
    doc.set_value('ext_Court', extractor.court(doc.title))
    doc.set_value('ext_Year', extractor.year(doc.title))

    doc.set_value('ext_Age', extractor.age(cleanText))
    doc.set_value('ext_AgeRange', extractor.ageRange(cleanText))
    doc.set_value('ext_Sentences', extractor.sentence(cleanText))
    #doc.set_value('ext_Victim', extractor.victimRelated(cleanText))
    doc.set_value('ext_Reconciliation', extractor.findWordlistElem(cleanText, 'reconciliation'))
    doc.set_value('ext_FamilyRelations', extractor.findWordlistElem(cleanText, 'family'))
    doc.set_value('ext_sentencingType', extractor.findWordlistElem(cleanText, 'sentencing'))

    doc.set_value('ext_CaseType', extractor.caseType(doc.text))
    
    entities = ner.getNamedEntities(doc.text)
    for entity in entities:
        doc.set_value(entity[0], entity[1])

    viewer = Viewer('FeatureExtraction')
    features = ['Court', 'Year', 'Age', 'ext_Court', 'ext_Year', 'ext_CaseType', 'ext_Age', 'ext_AgeRange', 'ext_Sentences', 'ORGANIZATION', 'LOCATION', 'PERSON', 'ext_Reconciliation', 'ext_FamilyRelations', 'ext_sentencingType']
    viewer.printDocument(doc, features, True)

    pdb.set_trace()



if __name__ == '__main__':
    FeatureExtraction_demo()
