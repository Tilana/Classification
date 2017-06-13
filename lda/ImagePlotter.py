import matplotlib.pyplot as plt
import numpy as np

def plotHistogram(data, title='', path=None, xlabel='', ylabel='', log=False, start=None, end=None, bins=10, open=False):
    if data != []:
        if start == None:
            start = min(data)
        if end == None:
            end = max(data)
        figure = plt.gcf()
        plt.hist(data, bins, range=(start,end), log=log)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if open:
            plt.show()
        plt.draw()
        if path:
            figure.savefig(path)
        plt.clf()


def boxplot(data):
    figure = plt.gcf()
    plt.boxplot(data)
    plt.show()


def barplot(data, title='', xlabel='', ylabel=''):
    figure = plt.gcf()
    index = np.arange(len(data))
    plt.barh(index, data, alpha=0.5)
    plt.yticks(index, ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

