import matplotlib.pyplot as plt

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

