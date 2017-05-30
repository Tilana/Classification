from lda import Viewer, FeatureExtractor, Preprocessor
import matplotlib.pyplot as plt
import pandas as pd
import pdb

def plotFrequency(data, colName):
    try:
        plt.clf()
        plt.subplot(1,1,1)
        data[colName].hist()
        plt.title('Histogram of '+ colname + '   N:' + str(len(data)))
        plt.ylabel('Frequency in Number of Documents')
        plt.xlabel(colName)
        plt.savefig('Plots/'+colName + '.jpg')
        plt.clf()
    except:
        pass



def FeatureAnalysis(data):

    for col in data.columns:
        plotFrequency(data, col)

    pdb.set_trace()
    caseType = data.ext_CaseType.value_counts(sort=False, dropna=False)
    ax = caseType.plot.barh(title='Document Type  N: '+str(len(data)))
    ax.set_xlabel('Number of Documents')
    plt.show()
    
    pdb.set_trace()

    nones = SAcases[SAcases.Type.isnull()]
    viewer = Viewer('FeatureExtraction')
    features = ['Court', 'Year', 'Age', 'ext_Court', 'ext_Year', 'ext_CaseType', 'ext_Age', 'ext_AgeRange', 'ext_Sentences', 'ORGANIZATION', 'LOCATION', 'PERSON', 'ext_Reconciliation', 'ext_FamilyRelations', 'ext_sentencingType']
    nones.apply(lambda doc: viewer.printDocument(doc, features, True), axis=1)



if __name__ == '__main__':
    FeatureAnalysis()
