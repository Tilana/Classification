import matplotlib.pyplot as plt

def plotHistogram(data, title, path, xlabel, ylabel, log, start=None, end=None, bins=None, open=0):
    if data != []:
        if bins == None:
            bins = 10
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
        figure.savefig(path)
        plt.clf()

