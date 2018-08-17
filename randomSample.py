from sentenceTokenizer import tokenize
from pymongo import MongoClient
import codecs
import json
import random
import numpy as np
import re

MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 4

def randomSample(collection):

    client = MongoClient('localhost', 27017)
    db = client[collection]['entities']

    sentences = []

    N_docs = 50
    N_sentences = 5

    count = db.count()
    for ind in range(0,N_docs):
        doc = db.find()[random.randrange(count)]

        text = doc['fullText']
        text = re.sub(r'\[\[[0-9]+\]\]', '', text)

        doc_sentences = tokenize(text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
        sample_indices = np.random.randint(0, len(doc_sentences), (N_sentences,))
        sample_sentences= [doc_sentences[sample_index] for sample_index in sample_indices]

        sentences.append(sample_sentences)

    sentences = sum(sentences, [])

    with codecs.open('random_sentences_echr.txt', 'w', encoding='utf8') as output_file:
        output_file.write(json.dumps(sentences, ensure_ascii=False))



if __name__=='__main__':
    randomSample('echr')

