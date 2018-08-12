import tensorflow as tf
import tensorflow_hub as hub
from lda import Preprocessor
from sentenceTokenizer import tokenize
import pandas as pd
import numpy as np
import time
import pdb

import sys
sys.path.append('../')

PATH = '../../data/ECHR/echr_docs.csv'
#PATH = '../data/ECHR/echr_fullText.json'

EVIDENCES = pd.read_csv('../data/ECHR/trainingSentences.csv', encoding='utf8')

MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 4
THRESHOLD = 0.65


sent_encoder_graph = tf.get_default_graph()
sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

sentences = tf.placeholder(dtype=tf.string, shape=[None])
embedding = sentenceEncoder(sentences)


def get_similar_sentences(similarity, sentences):
    evidenceIndices, sentenceIndices = np.where(similarity >= THRESHOLD)
    similarSentences = [(similarity[evidenceInd, sentenceInd], sentences[sentenceInd]) for (evidenceInd, sentenceInd) in zip(evidenceIndices, sentenceIndices)]
    return similarSentences


def predictDoc(doc, sess, evidence_embedding):
    doc_sentences = tokenize(doc['text'], MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
    sentence_embedding = sess.run(embedding, feed_dict={sentences: doc_sentences})
    similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))

    pdb.set_trace()
    return get_similar_sentences(similarity, doc_sentences)

for index, evidence in EVIDENCES.iterrows():
    print 'ID: ' + str(evidence.ID)
    t0 = time.time()

    data = pd.read_csv(PATH, index_col=0, encoding='utf8')
    with sent_encoder_graph.as_default():
        with tf.Session(graph=sent_encoder_graph) as session:

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])


            data = data[data.title.str.contains('BAKA')]

            evidence_embedding = session.run(embedding, feed_dict={sentences: [evidence['sample expression']]})
            data['similarSentences'] = data.apply(predictDoc, sess=session, evidence_embedding=evidence_embedding, axis=1)

            session.close()
    print 'Time: {} \n'.format(time.time()-t0)

    data.drop(columns=['text'], inplace=True)
    data['nrSentences'] = data.similarSentences.map(len)
    data = data[data.nrSentences !=0]
    data.to_csv('../data/ECHR/suggestions_{}_{}2.csv'.format(evidence.ID, THRESHOLD), encoding='utf8', index=False)


