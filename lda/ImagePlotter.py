import matplotlib.pyplot as plt
import seaborn as sns
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
        #plt.hist(data, bins+1, range=(start,end), log=log)
        plt.hist(data, bins, range=(start,end), log=log)
        self.labelFigure(title, xlabel, ylabel)
        self.showFigure()
        self.save(path)
        self.closeFigure()

    def heatmap(self, data, path, xlabel, ylabel):
        self.createFigure()
        ax = sns.heatmap(data.T, cmap='Blues')
        ax.set_xticks(np.arange(len(xlabel)))
        ax.set_yticks(np.arange(len(ylabel)))

        ax.set_xticklabels(xlabel)
        ax.set_yticklabels(ylabel)

        for item in ax.get_yticklabels():
            item.set_rotation(0)
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        self.showFigure()
        self.save(path)

    def boxplot(self, data):
        self.createFigure()
        plt.boxplot(data, showmeans=1, meanline=1)
        self.showFigure()
        self.closeFigure()


    def barplot(self, data, title='', xlabel='', ylabel='', xticks=None, path='', log=False):
        self.createFigure()
        if not xticks:
            xticks = np.arange(len(data))
        plt.barh(xticks, data, alpha=0.5, log=log)
        plt.yticks(xticks, ylabel)
        self.labelFigure(title, xlabel, ylabel='')
        self.showFigure()
        self.save(path)
        self.closeFigure()

    def multiBar(self, data1, data2, title='', xlabel='', ylabel='', xticks=None, path='', log=False):
        self.createFigure()
        if not xticks:
            xticks = np.arange(len(data1))
        plt.bar(xticks, data1, color='g')



    def createFigure(self):
        self.figure = plt.gcf()


    def closeFigure(self):
        plt.clf()


    def save(self, path):
        try:
            self.figure.savefig(path)
            #self.figure.savefig(path, bbox_inches='tight')
        except:
            print('Error ImagePlotter: Figure cannot be saved')

    def showFigure(self):
        if self.show:
            plt.show()

    def labelFigure(self, title='', xlabel='', ylabel=''):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)



