from lda import Viewer, FeatureExtractor
import pandas as pd
import numpy as np

def FeatureExtraction_demo():

    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)
    
    ind = 51
    title = data.loc[ind, 'title']
    text = data.loc[ind, 'text']

    extractor = FeatureExtractor()
    data.loc[ind,'extCourt']  = extractor.court(title)
    data.loc[ind,'extYear'] = extractor.year(title)

    print extractor.age(text)
    print type(extractor.age(text))
    data.loc[ind,'extAge'] = [extractor.age(text)]
    ##data.loc[ind,'extSentences'] = extractor.sentence(text)
    data.loc[ind,'extCaseType'] = extractor.caseType(text)

    doc = data.loc[ind]
    print doc.keys()

    viewer = Viewer('FeatureExtraction')
    features = ['Court', 'Year', 'extCourt', 'extYear', 'extCaseType']
    viewer.printDocument(doc, features, True)



if __name__ == '__main__':
    FeatureExtraction_demo()
