import matplotlib.pyplot as plt
import numpy as np
import pdb

class ImagePlotter:

    def __init__(self, show=False):
        self.show = show 

    
    def plotHistogram(self, data, title='', path=None, xlabel='', ylabel='', log=False, start=None, end=None, bins=10, open=False, replaceNAN=False):
        if replaceNAN:
            data = [elem if not np.isnan(elem) else -1 for elem in data]
        else:
            data = [elem for elem in data if not np.isnan(elem)]
        if start == None:
            start = min(data)
        if end == None:
            end = max(data)
        self.createFigure()
        plt.hist(data, bins+1, range=(start,end), log=log)
        self.labelFigure(title, xlabel, ylabel)
        self.showFigure()
        self.save(path)
        self.closeFigure()
    
    
    def boxplot(self, data):
        self.createFigure() 
        plt.boxplot(data)
        self.showFigure()
        self.closeFigure()
    
    
    def barplot(self, data, title='', xlabel='', ylabel='', xticks=None, path=''):
        self.createFigure()
        if not xticks:
            xticks = np.arange(len(data))
        plt.barh(xticks, data, alpha=0.5)
        plt.yticks(xticks, ylabel)
        self.labelFigure(title, xlabel, ylabel='')
        self.showFigure()
        self.save(path)
        self.closeFigure()


    def createFigure(self):
        self.figure = plt.gcf()


    def closeFigure(self):
        plt.clf()


    def save(self, path):
        try:
            self.figure.savefig(path, bbox_inches='tight')
        except:
            print 'Error ImagePlotter: Figure cannot be saved'

    def showFigure(self):
        if self.show:
            plt.show()

    def labelFigure(self, title='', xlabel='', ylabel=''):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


    
