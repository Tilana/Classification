import tensorflow as tf
import tensorflow_hub as hub
from sentenceTokenizer import tokenize
from pymongo import MongoClient
import pandas as pd
import numpy as np
import time
import re


MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 4
THRESHOLD = 0.65

sent_encoder_graph = tf.get_default_graph()
sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

sentences = tf.placeholder(dtype=tf.string, shape=[None])
embedding = sentenceEncoder(sentences)


def similarSentences(collection, category, evidences, method='avg'):

    client = MongoClient('localhost', 27017)
    db = client[collection]['entities']

    suggestions = []
    t0 = time.time()

    with sent_encoder_graph.as_default():
        with tf.Session(graph=sent_encoder_graph) as session:

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            evidence_embedding = session.run(embedding, feed_dict={sentences: evidences})

            if method=='vec_avg':
                evidence_embedding = np.average(evidence_embedding, axis=0)
                evidence_embedding = evidence_embedding.reshape(1,evidence_embedding.shape[0])

            for doc in db.find():
                text = doc['fullText']
                text = re.sub(r'\[\[[0-9]+\]\]', '', text)

                doc_sentences = tokenize(text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
                sentence_embedding = session.run(embedding, feed_dict={sentences: doc_sentences})

                similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))

                if method=='avg':
                    similarity = np.average(similarity, axis=0)
                    similarity = similarity.reshape(1, similarity.shape[0])

                evidenceIndices, sentenceIndices = np.where(similarity >= THRESHOLD)
                similar_sentences = [(similarity[evidenceInd, sentenceInd], doc_sentences[sentenceInd]) for (evidenceInd, sentenceInd) in zip(evidenceIndices, sentenceIndices)]

                if similar_sentences != []:

                    avg_evidence_similarity = np.mean(zip(*similar_sentences)[0])
                    avg_doc_similarity = np.average(np.max(similarity, axis=0),axis=0)

                    similar_sentences = pd.DataFrame(similar_sentences, columns=['probability', 'sentences'])
                    similar_sentences.sort_values('probability', inplace=True, ascending=False)
                    similar_sentences.drop_duplicates('sentences', inplace=True)

                    similar_sentences = [tuple(sentence) for sentence in similar_sentences.to_records(index=False)]

                    suggestions.append((doc['sharedId'], doc['title'], avg_doc_similarity,avg_evidence_similarity, len(similar_sentences), similar_sentences))

                print('ID: {}  -  Title: {}'.format(doc['sharedId'], doc['title'].encode('utf8')))

            session.close()
    print 'Time: {} \n'.format(time.time()-t0)

    suggestions = pd.DataFrame(suggestions, columns=['doc_id', 'title', 'avg_doc_similarity', 'avg_similarity', 'nr_evidences', 'evidences'])
    suggestions.sort_values(['nr_evidences', 'avg_similarity'], inplace=True, ascending=False)
    suggestions.to_csv('similar_sentences/{}/{}_{}.csv'.format(collection, category, method), index=False, encoding='utf8')



if __name__=='__main__':

    methods = ['all']#, 'avg'] #, 'vec_avg']
    already_processed = ['data_protection_short', 'data_protection_long']

    all_evidences = pd.read_csv('trainingSentences_multiple.csv', encoding='utf8')
    evidence_cols = [col for col in all_evidences.columns if 'evidence' in col]

    for index, row in all_evidences.iterrows():
        category = row['ID']

        if category not in already_processed:
            print category

            evidences = row[evidence_cols].tolist()

            for method in methods:
                print 'METHOD: {}'.format(method)
                similarSentences('echr', category, evidences, method=method)

