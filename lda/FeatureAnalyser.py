from lda import Viewer 
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from lda import listUtils as utils
from ImagePlotter import ImagePlotter

class FeatureAnalyser:

    def __init__(self):
        self.plotter = ImagePlotter()


    def relevantWords(self, collection, target):
        targetData = collection.data[target].tolist()
        data = collection.getFeatureList(collection.data, 'tfIdf')

        bestFeatures = collection.FeatureSelection(data, targetData)
        scores = zip(collection.vocabulary, collection.FeatureSelector.scores_)

        sortedScores = utils.sortTupleList(scores)
        words, scores = zip(*sortedScores)

        n = 50
        barplot(scores[:n], target + ' - Chi-Square relevant words', 'Chi-Square score', words[:n])


    def frequencyPlots(self, collection):
        for col in collection.data.columns:
            data = collection.data[col]
            values = data.value_counts(sort=False)
            if len(values) < 50:
                self.plotter.barplot(values.tolist(), ylabel=values.index, path='Plots/'+collection.name+'/'+col+'.jpg', title=col, xlabel='Number of Documents')


    def correlateVariables(self, collection): 
        correlationMatrix = collection.data.corr()
        self.plotter.heatmap(correlationMatrix, path='Plots/'+collection.name+'/'+'featureCorrelation_countries2.jpg')
        return correlationMatrix 


