from lda import Viewer 
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from lda import listUtils as utils
from lda.ImagePlotter import barplot, plotHistogram


def FeatureAnalysis(collection, target=None):

    if target:
        targetData = collection.data[target].tolist()
        data = collection.getFeatureList(collection.data, 'tfIdf')

        bestFeatures = collection.FeatureSelection(data, targetData)
        scores = zip(collection.vocabulary, collection.FeatureSelector.scores_)

        sortedScores = utils.sortTupleList(scores)
        words, scores = zip(*sortedScores)

        n = 50
        barplot(scores[:n], target + ' - Chi-Square relevant words', 'Chi-Square score', words[:n])

    #pdb.set_trace()
    #collection.data = collection.data.fillna(-1)
    print 'Histograms'
    for col in collection.data.columns:
        plotHistogram(collection.data[col].tolist(), path='Plots/HRC/'+col+'.jpg', title=col, ylabel='Number of Documents', xlabel=col)

    
    print 'Correlation of Variables'
    collection.data.corr()

    pdb.set_trace()
    caseType = collection.data.ext_CaseType.value_counts(sort=False, dropna=False)
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
