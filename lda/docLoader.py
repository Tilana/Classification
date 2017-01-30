import json
import urllib
import pickle
import os
import pandas


def loadCouchdb(path):
    dat = json.load(urllib.urlopen(path))
    rows = dat['rows']
    docs = list(map((lambda doc: doc['value']['Text']),rows))
    titles = list(map((lambda doc: doc['value']['Title']),rows))
    return (titles, docs)


def loadTxtFiles(path):
    titles = [txtfile for txtfile in os.listdir(path)]
    docs = [open(path+'/'+txtfile).read() for txtfile in os.listdir(path)]
    return (titles, docs)

def loadEncodedFiles(path):
    titles, texts = loadTxtFiles(path)
    titles = [removeSpecialChars(title) for title in titles]
    texts = [removeSpecialChars(text) for text in texts]
    return (titles, texts)



def loadCsvFile(path):
    data = pandas.read_csv(path) #, encoding='utf8')
    titles = list(data['Title'])
#    docs = list(data['Abstract'])
    docs = list(data['PaperText'])
    titles = [removeSpecialChars(title) for title in titles]
    docs = [removeSpecialChars(text) for text in docs]
    return (titles, docs)


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

