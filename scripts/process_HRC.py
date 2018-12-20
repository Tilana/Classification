import sys
import bson
sys.path.append('../')
import pandas as pd
import re
from bs4 import BeautifulSoup
from sentenceTokenizer import tokenize
import pdb
import jsonlines
import demjson

from bson.json_util import loads


def removePageNumbers(text):
    return re.sub(r'\[\[[0-9]+\]\]', '', text)


def getFulltextFromPages(pages):
    fulltext = []
    for page_num in sorted(pages.keys()):
        fulltext.append(pages[page_num])
    return ' '.join(fulltext)


def preprocess(sentence):
    result = re.sub('[^a-zA-Z0-9 ]+', ' ', sentence)
    #result = sentence.replace(",", " , ")
    #result = result.replace(";", " ; ")
    #result = result.replace("'", " ' ")
    #result = result.replace("'", " ' ")
    #result = result.replace('"', ' " ')
    #result = result.replace(":", " : ")
    #result = result.replace(".", " . ")
    #result = result.replace("(", " ( ")
    #result = result.replace(")", " ) ")
    #result = result.replace("[", " [ ")
    #result = result.replace("]", " ] ")
    return result.lower()


def generateValidJSON(text):
    text = re.sub(r'ObjectId\("[a-z0-9]*"\)', '"default"', text)
    text = re.sub(r'NumberInt\([0-9]*\)', '"default"', text)
    text = re.sub(r'\n\}\n', '}$$$$', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\$\$\$\$', '\n', text)
    #text = text.replace('ObjectId(', '')
    #text = text.replace('NumberInt(', '')
    #text = text.replace(')', '')
    return text

#PATH = '../../data/HRC/HRC.csv'
#PATH = '../../data/ICAAD/icaad.json'
#PATH = '../../data/ECHR/echr.json'
PATH = '../../data/HRC/rightdocs/entities.json'
PATH = '../../data/HRC/rightdocs/valid_entities.json'



#with jsonlines.open(VALID_JSON_PATH) as data:
#    #pdb.set_trace()
#    for text in data:
#        print text['title']
#        #pdb.set_trace()
#pdb.set_trace()
#
#with open(PATH, 'rb') as f:
#    text = generateValidJSON(f.read())
#    f.close()
#
##pdb.set_trace()
#
#with open(VALID_JSON_PATH, 'wb') as f:
#    f.write(text)
#    f.close()
#
#pdb.set_trace()

#with open(PATH, 'rb') as f:
#    text = f.read()
#    text2 = re.sub(r'\n\}\n', '}$$$$', text)
#    text2 = re.sub(r'\n', '', text2)
#    text2 = re.sub(r'\$\$\$\$', '\n', text2)

#with open('test.json', 'wb') as f:
#    f.write(text2)
#    .replace('\n', '')
#    pdb.set_trace()
#    bson.BSON.decode(f)
#    #bson.decode(text)
#    #data = demjson.decode(f.read())
#    data = bson.decode_file_iter(open(PATH, 'rb'))
#    #data = bson.decode(f.read())
#    data = bson.decode_all(f.read())
#    data = bson.decode_all(text)
#    df = pd.read_csv(f)
#    #data = pd.read_json(f)

texts = []
with jsonlines.open(PATH, 'r') as data:
    for text in data:
        if text.haskey('fullText'):
            fulltext = getFulltextFromPages(text['fullText'])
            clean_text = removePageNumbers(fulltext)
            texts.append(clean_text)

#pdb.set_trace()

# texts = [text for text in df.text.tolist() if isinstance(text, str)]
#all_texts = '\n \n'.join(texts)
all_texts = ' '.join(texts)

# clean_texts = BeautifulSoup(all_texts, 'lxml').text
#clean_texts = re.sub('(\n)+', ' ', all_texts)
#clean_texts = preprocess(clean_texts)


#sentences = tokenize(clean_texts)
#sentences = '\n'.join(sentences)

#pdb.set_trace()

#clean_texts = preprocess(sentences)
#clean_texts = preprocess(clean_texts)
clean_texts = preprocess(all_texts)
clean_texts = ' '.join(clean_texts.split())

fs = open('hrc.txt', 'wb')
#fs.write(sentences.encode('utf8')) #, encode='utf8')
fs.write(clean_texts.encode('utf8')) #, encode='utf8')
fs.close()
