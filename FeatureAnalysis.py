from lda import Viewer, FeatureExtractor, Preprocessor
import matplotlib.pyplot as plt
import pandas as pd
import pdb

def FeatureAnalysis():

    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)
    targets = ['Domestic.Violence.Manual', 'Sexual.Assault.Manual']
    target = targets[1] 
    
    indexSA = data[data[target]].index.tolist()
    indexSA = data[data[target]].index.tolist()

    SAcases = data.loc[indexSA]

    extractor = FeatureExtractor()
    SAcases['Type'] = SAcases.apply(lambda doc: extractor.caseType(doc.text), axis=1)

    plt.figure()
    caseType = SAcases.Type.value_counts(sort=False, dropna=False)
    ax = caseType.plot.barh(title=target + '  N: '+str(len(SAcases)))
    ax.set_xlabel('Number of Documents')
    plt.show()
    pdb.set_trace()

    nones = SAcases[SAcases.Type.isnull()]

    
    #pdb.set_trace()

    viewer = Viewer('FeatureExtraction')
    features = ['Court', 'Year', 'Age', 'ext_Court', 'ext_Year', 'ext_CaseType', 'ext_Age', 'ext_AgeRange', 'ext_Sentences', 'ORGANIZATION', 'LOCATION', 'PERSON', 'ext_Reconciliation', 'ext_FamilyRelations', 'ext_sentencingType']

    #nones.apply(lambda doc: viewer.printDocument(doc, features, True), axis=1)

    #pdb.set_trace()





if __name__ == '__main__':
    FeatureAnalysis()
