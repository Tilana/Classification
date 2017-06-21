import json
import urllib
import pickle
import os
import pandas as pd


def loadData(path):
    fileFunctionMatch = {'csv': loadCSV, 'pkl': loadPickle}
    fileEnding = path.split('.')[-1]
    loadFunction = fileFunctionMatch[fileEnding]
    return loadFunction(path)
    

def loadCouchDB(path):
    dat = json.load(urllib.urlopen(path))
    rows = dat['rows']
    docs = list(map((lambda doc: doc['value']['Text']),rows))
    titles = list(map((lambda doc: doc['value']['Title']),rows))
    return (titles, docs)


def loadTXTFolder(path):
    titles = [txtfile for txtfile in os.listdir(path)]
    docs = [open(path+'/'+txtfile).read() for txtfile in os.listdir(path)]
    return (titles, docs)


def loadEncodedFiles(path):
    titles, texts = loadTxtFiles(path)
    titles = [removeSpecialChars(title) for title in titles]
    texts = [removeSpecialChars(text) for text in texts]
    return (titles, texts)


def loadCSV(path):
    return pd.read_csv(path)


def loadPickle(path):
    return pd.read_pickle(path)


def loadCategories(path):
    f = open(path, 'r')
    categories = f.readlines()
    f.close()
    return [wordlist.split() for wordlist in categories]


def removeSpecialChars(text, verbosity=0):
    encodedText = []
    for word in text.split():
        try:
            encodedWord = word.encode('utf8')
            encodedText.append(encodedWord)
        except:
            if verbosity:
                print "Failed Encoding: ", word
            pass
    return " ".join(encodedText)


def storeAsTxt(dat, path):
    with open(path, 'wb') as f:
        pickle.dump(dat, f)


def loadTxt(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

