from lda import Viewer, FeatureExtractor, Preprocessor
from lda import namedEntityRecognition as ner
import pandas as pd
import numpy as np
import pdb

def FeatureExtraction_demo():

    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)
    
    ind = 51
    title = data.loc[ind, 'title']
    text = data.loc[ind, 'text']

    extractor = FeatureExtractor()
    processor = Preprocessor()
    posTags = processor.posTagging(processor.wordTokenize(text.lower()))
    lemmas = processor.posLemmatize(posTags)
    cleanText =  ' '.join(lemmas)
    
    data.loc[ind,'extCourt']  = extractor.court(title)
    data.loc[ind,'extYear'] = extractor.year(title)

    data['extAge'] = 's'
    data['extSentences'] = 's'
    data['victim'] = 's'
    data['ORGANIZATION'] = 's'
    data['LOCATION'] = 's'
    data['PERSON'] = 's'
    data.set_value(ind, 'extAge', extractor.age(cleanText))
    data.set_value(ind, 'extSentences', extractor.sentence(cleanText))
    data.loc[ind,'extCaseType'] = extractor.caseType(text)
    #pdb.set_trace()
    print extractor.victimRelated(cleanText)
    data.set_value(ind,'victim', extractor.victimRelated(cleanText))
    print extractor.accusedRelated(cleanText)
    
    print extractor.victimRelated(text)
    print extractor.accusedRelated(text)

    entities = ner.getNamedEntities(text)
    for entity in entities:
        data.set_value(ind, entity[0], entity[1])

    pdb.set_trace()



    doc = data.loc[ind]

    viewer = Viewer('FeatureExtraction')
    features = ['Court', 'Year', 'Age', 'extCourt', 'extYear', 'extCaseType', 'extAge', 'extSentences', 'victim']
    viewer.printDocument(doc, features, False)



if __name__ == '__main__':
    FeatureExtraction_demo()
