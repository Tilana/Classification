from __future__ import division
import sys
sys.path.append('../InferSent')
from models import InferSent
import torch
from lda import Evaluation, Viewer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import re
import pdb
import fastText

PATH = '../data/ICAAD/ICAAD_labeledSentences.csv'
STOPWORDS = ['of', 'and', 'then', 'by', 'for', 'a', 'the', 'from', 'that', 'to', 'with', 'within', 'this', 'so', 'as', 'on', 'in', 'therefore', 'is', 'it', 'that', 'at']
FREQUENT_WORDS = ['court', 'judge', 'counsel', 'offence', 'plaintiff', 'accused', 'defendant', 'sentence', 'appeal', 'commit', 'committed', 'charge', 'charged', 'unlawful', 'record']
EXCLUDE = STOPWORDS + FREQUENT_WORDS

ENCODING = 'fasttext'
#ENCODING = 'USE'
#ENCODING = 'InferSent'

THRESHOLD = 0.45
logit = 0
a = 30.0
rm_stopwords = 0

InferSent_path = '../InferSent/encoder/infersent1.pkl'
W2V_PATH = '../InferSent/dataset/fastText/crawl-300d-2M.vec'
W2V_PATH = '../InferSent/dataset/GloVe/glove.840B.300d.txt'
W2V_PATH = '../fastText/model.bin'
W2V_PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.bin'
params_model = {'bsize': 64, 'word_emb_dim':300, 'enc_lstm_dim':2048, 'pool_type':'max', 'dpout_model':0.0, 'version':2}


model = fastText.load_model(W2V_PATH)
#model = InferSent(params_model)
#model.load_state_dict(torch.load(InferSent_path))
#model.set_w2v_path(W2V_PATH)
#model.build_vocab_k_words(K=100000)

#pdb.set_trace()

sent_encoder_graph = tf.get_default_graph()
#sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
# sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
#sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

#sentences = tf.placeholder(dtype=tf.string, shape=[None])
#embedding = sentenceEncoder(sentences)

def cosine(u, v):
    return np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))

def preprocess(sentence):
    result = sentence.replace(",", " , ")
    result = result.replace(":", " : ")
    result = result.replace(";", " ; ")
    return result


def removeStopwords(sentence):
    tokens = sentence.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

def transformProbability(p,a):
    return p**a/(p**a + (1-p)**a)

def relu(p):
    if p<0:
        return 0
    else:
        return p

def avg_word_vectors(sentence):
    vec = [model.get_word_vector(word) for word in sentence.split()]
    weights = [1 if word not in EXCLUDE else 0.5 for word in sentence.split()]
    return np.average(np.array(vec), weights=weights, axis=0)


def ICAAD_classification_use(data, category, config, evidences, method='max'):

    print('** EMBEDDING **')
    t0 = time.time()

    test_sentences = data.sentence.tolist()
    data['targetLabel'] = data.category == category

    if ENCODING=='USE':
        with sent_encoder_graph.as_default():
            with tf.Session(graph=sent_encoder_graph) as session:

                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                evidence_embedding = session.run(embedding, feed_dict={sentences: evidences})

                if rm_stopwords:
                    test_sentences = [removeStopwords(sentence) for sentence in test_sentences]
                    data['sentence'] = test_sentences
                sentence_embedding = session.run(embedding, feed_dict={sentences: test_sentences})

                similarity = np.inner(evidence_embedding, sentence_embedding)

                #cos = tf.reduce_sum(tf.multiply(evidence_embedding, sentence_embedding), axis=1)
                #clip_cos = tf.clip_by_value(cos, -1.0, 1.0)
                #scores = 1.0 - tf.acos(clip_cos)

                #if logit:
                #    pos = np.vectorize(relu)
                #    similarity = pos(similarity)
                #    f = np.vectorize(transformProbability)
                #    similarity = f(similarity, a)

                #if method=='max':
                #    similarity = np.max(similarity, axis=0)
                #    similarity = similarity.reshape(1, similarity.shape[0])

                #if method=='avg':
                #    similarity = np.average(similarity, axis=0)
                #    similarity = similarity.reshape(1, similarity.shape[0])

                #data['similarity'] = similarity.reshape(len(data))
                #data['similarity'] = data.similarity.apply(round, ndigits=4)

                session.close()

    elif ENCODING=='fasttext':

        test_sentences = [preprocess(sentence) for sentence in test_sentences]
        evidences = [preprocess(evidence) for evidence in evidences]

        sentence_embedding = [avg_word_vectors(sentence) for sentence in test_sentences]
        evidence_embedding = [avg_word_vectors(evidence) for evidence in evidences]

        #evidence_embedding = [model.get_sentence_vector(preprocess(evidence)) for evidence in evidences]
        #sentence_embedding = [model.get_sentence_vector(preprocess(sentence)) for sentence in test_sentences]

        #similarity2 = np.inner(evidence_embedding, sentence_embedding)
        similarity = []
        for evd_embed in evidence_embedding:
            for sent_emb in sentence_embedding:
                similarity.append(cosine(evd_embed, sent_emb))
        similarity = np.array(similarity).reshape((len(evidence_embedding), len(sentence_embedding)))

        #pdb.set_trace()

    elif ENCODING=='InferSent':
        #pdb.set_trace()
        #model.encode(test_sentences)
        #test_sentences = test_sentences[:20]

        model.build_vocab(test_sentences, tokenize=True)

        evidence_embedding = model.encode(evidences, bsize=128, tokenize=True, verbose=True)
        sentence_embedding = model.encode(test_sentences, bsize=128, tokenize=True, verbose=True)

        similarity = []
        for evd_embed in evidence_embedding:
            for sent_emb in sentence_embedding:
                similarity.append(cosine(evd_embed, sent_emb))
        similarity = np.array(similarity).reshape((len(evidence_embedding), len(sentence_embedding)))


    if method=='logreg':
        for num in range(similarity.shape[0]):
            data['sim_' + str(num)] = similarity[num, :]
        data['avg'] = data[['sim_0', 'sim_1']].mean(axis=1)
        data['predictedLabel'] = data['avg'] >= THRESHOLD

        certainCases = (data['avg']>=0.7) | (data['avg']<0.2)
        train = data[certainCases]
        test = data[~certainCases]

        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        logreg.fit(train[['sim_0', 'sim_1']], train['predictedLabel'])
        test['predictedLabel'] = logreg.predict(test[['sim_0', 'sim_1']])
        test['similarity'] = logreg.predict_log_proba(test[['sim_0', 'sim_1']])[:,1]

        data = test

    if logit:
        pos = np.vectorize(relu)
        similarity = pos(similarity)
        f = np.vectorize(transformProbability)
        similarity = f(similarity, a)

    if method=='max':
        similarity = np.max(similarity, axis=0)
        similarity = similarity.reshape(1, similarity.shape[0])

    if method=='avg':
        similarity = np.average(similarity, axis=0)
        similarity = similarity.reshape(1, similarity.shape[0])

    if method=='max' or method=='avg':
        data['similarity'] = similarity.reshape(len(data))
        data['similarity'] = data.similarity.apply(round, ndigits=4)

        print('USE Time: {}'.format(time.time()-t0))

        print('** EVALUATION **')
        t0 = time.time()

        data['predictedLabel'] = data.similarity >= THRESHOLD

    evaluation = Evaluation(target=data.targetLabel.tolist(), prediction=data.predictedLabel.tolist())
    evaluation.computeMeasures()
    evaluation.confusionMatrix()

    print('Evaluation Time: {}'.format(time.time()-t0))

    print('Accuracy: ' + str(evaluation.accuracy))
    print('Recall: ' + str(evaluation.recall))
    print('Precision: ' + str(evaluation.precision))
    print(evaluation.confusionMatrix)

    evaluation.createTags()
    data['tags'] = evaluation.tags
    data.sort_values(['tags', 'similarity'], inplace=True, ascending=[True, False])
    pd.set_option('display.max_colwidth', 500)

    html_name = config + '_' + ENCODING
    if len(evidences)>1:
        html_name = html_name + '_' + method
    if rm_stopwords:
        html_name = html_name + '_rmStopwords'
    if logit:
        html_name = html_name + '_logit'

    axes = data.hist('similarity')
    plt.savefig(html_name + '_prediction.png')

    axes = data.hist('similarity', by='tags')
    plt.savefig(html_name + '_tags.png')

    viewer = Viewer('use')
    viewer.use_classificationResults(html_name, evidences, data.drop(columns=['id', 'category', 'predictedLabel', 'targetLabel']), THRESHOLD, evaluation, a)



if __name__=='__main__':

    data = pd.read_csv(PATH, encoding='utf8')
    #data = data.sample(1000, random_state=42)

    category = 'Evidence.of.SA'
    negCategory = 'Evidence.no.SADV'
    data = data[(data.category==category) | (data.category==negCategory)]

    #data.sentence = data.sentence[:21]
    #data = data.sample(100, random_state=42)

    #set_evidences = [('keyword_rape', ['rape']), ('keyword_sexual_assault', ['sexual assault']), ('multiple_keywords', ['rape', 'sexual assault']), ('multiple_keywords_string', ['rape sexual assault']), ('multiple_keywords_string2', ['sexual assault rape'])]
    # set_evidences = [('keyword_rape', ['rape']), ('phrase_rape', ['rape is committed when having sexual intercourse without consent']), ('combined', ['rape', 'rape is committed when having sexual intercourse without consent'])]
    # set_evidences = [('combined', ['rape', 'rape is committed when having sexual intercourse without consent', 'sexual assault', 'incest', 'indecent assault', 'carnal knowledge'])]
    #set_evidences = [('phrase_rape', ['rape is committed when having sexual intercourse without consent'])]
    set_evidences = [('keyword_rape', ['rape'])]

    for config, evidences in set_evidences:
        ICAAD_classification_use(data, category, config, evidences, method='max')

