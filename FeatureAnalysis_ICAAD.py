import pandas as pd
import numpy as np
from matplotlib import cm
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

def FeatureAnalysis_ICAAD():

    evaluationFile = 'Documents/PACI.csv'
    evaluationFeatures = pd.read_csv(evaluationFile)
    evaluationFeatures = evaluationFeatures.rename(columns={'Unnamed: 0': 'id'})
    featurePath = 'html/ICAAD_LDA_T60P10I70_tfidf_word2vec/'
    featureFile = featurePath + 'DocumentFeatures.csv'
    featureData = pd.read_csv(featureFile)

    data = pd.merge(featureData, evaluationFeatures, on=['id'])

    topicList = [('Topic%d' % topicNr) for topicNr in range(0,60)]
    similarDocList = [('similarDocs%d' % docNr) for docNr in range(1,6)]

    droplist = topicList + similarDocList 

    subData = data[['Topic12', 'Topic51', 'Topic47', 'Topic34', 'Topic33', 'Sexual.Assault.Manual']]
    subData = data[['rape', 'carnal knowledge', 'intercourse', 'indecent assault', 'penetration', 'sex', 'consent', 'Family.Member.Victim', 'Sexual.Assault.Manual']]

    subData.hist()
    #plt.hist(subData['domestic'], log=True)

    groupedData = subData.groupby('Sexual.Assault.Manual')
    .hist(log=True)

    scatter_matrix(subData, alpha=0.2, diagonal='kde')

    scatter_matrix(data[['Topic12', 'Topic51', 'Topic32', 'rape', 'Sexual.Assault.Manual']])

    t = data[['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Rape', 'Family.Member.Victim', 'rape', 'incest', 'family', 'stepfather', 'vagina']]
    plt.matshow(t.corr())

    data = data.drop(['Sexual.Assault', 'Domestic.Violence'], axis=1)
    dataCorrelation = data.corr()
    np.fill_diagonal(dataCorrelation.values, 0)
    dataCorrelation.describe()

    threshold = 0.7

    lowCorrelationFeatureNames = dataCorrelation[dataCorrelation.abs().max() < threshold].index.tolist()
    lowCorrelationFeatures = dataCorrelation.abs().max()[lowCorrelationFeatureNames].to_frame()
    lowCorrelationFeatures['MaxCorrFeature'] = dataCorrelation.abs().idxmax()[lowCorrelationFeatureNames]

    aboveThreshold = dataCorrelation.drop(lowCorrelationFeatureNames, axis=0)
    aboveThreshold = aboveThreshold.drop(lowCorrelationFeatureNames, axis=1)

    plotCorrelationMatrix(aboveThreshold)
    
    selectedFeatures = ['rape', 'carnal knowledge', 'sex', 'indecent assault', 'intercourse', 'stepfather', 'stepdaughter', 'Domestic.Violence.Manual', 'manslaughter', 'Family.Member.Victim']
    
    for feature in selectedFeatures:
        plotHistogramm(data, 'Sexual.Assault.Manual', feature)
    
    

    def plotCorrelationMatrix(dataframe):
        correlation = dataframe.corr()
        fig, ax = plt.subplots(figsize=(10,10))
        cax = ax.matshow(correlation)
        colorbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        #colorbar.ax.set_yticklabels(['< -1', '0', '>1'])
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
        plt.yticks(range(len(correlation.columns)), correlation.columns)


    def plotHistogramm(data, groupby, column):
        fig = plt.figure()
        for value in data[groupby].unique():
            datapoints = data[data[groupby]==value][column]
            plt.hist(datapoints, alpha=0.5, bins=range(0,20), label=str(value), log=True)
        plt.legend(loc='upper right')
        plt.title(groupby + ' - ' + column)
        plt.xlabel('Occurence of ' + column)
        plt.ylabel('Number of Documents')
        plt.savefig(featurePath + 'FeatureAnalysis/'+column + '.jpeg')
        plt.show()



if __name__ == "__main__":
    FeatureAnalysis_ICAAD()
