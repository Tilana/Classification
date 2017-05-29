from lda import Viewer, FeatureExtractor, Preprocessor
from lda import namedEntityRecognition as ner
import pandas as pd
import pdb

def applyToRows(data, field, fun, name):
    data[name] = data.apply(lambda doc: fun(doc[field]), axis=1)

def extractEntities(doc):
    entities = ner.getNamedEntities(doc.text)
    for entity in entities:
        doc.set_value(entity[0], entity[1])

def FeatureExtraction(data):

    data = data[data['Domestic.Violence.Manual']]
    
    preprocessor = Preprocessor()

    
    applyToRows(data, 'text', preprocessor.cleanText, 'cleanText') 
    applyToRows(data, 'cleanText', preprocessor.numbersInTextToDigits, 'cleanText') 
    
    extractor = FeatureExtractor()

    applyToRows(data, 'title', extractor.court, 'ext_Court') 
    applyToRows(data, 'title', extractor.year, 'ext_Year') 
    applyToRows(data, 'cleanText', extractor.age, 'ext_Age') 
    applyToRows(data, 'cleanText', extractor.ageRange, 'ext_AgeRange') 
    applyToRows(data, 'cleanText', extractor.sentence, 'ext_Sentences') 
    applyToRows(data, 'text', extractor.caseType, 'ext_CaseType') 
    pdb.set_trace()
    data.apply(extractEntities, axis=1)

    return data
    



if __name__ == '__main__':
    FeatureExtraction()
