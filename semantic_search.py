from lda import Evaluation, Viewer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import pdb
import fastText
from nltk.stem import PorterStemmer

PATH = '../data/ICAAD/ICAAD_labeledSentences.csv'
STOPWORDS = ['of', 'and', 'then', 'by', 'for', 'a', 'the', 'from', 'that', 'to', 'with', 'within', 'this', 'so', 'as', 'on', 'in', 'therefore', 'is', 'it', 'that', 'at']
FREQUENT_WORDS = ['court', 'judge', 'counsel', 'offence', 'plaintiff', 'accused', 'defendant', 'sentence', 'appeal', 'commit', 'committed', 'charge', 'charged', 'unlawful', 'record', 'complainant', 'count', 'appellant', 'offence', 'guilty']
EXCLUDE = STOPWORDS + FREQUENT_WORDS

ENCODING = 'fasttext'
# ENCODING = 'USE'
#we_model = 'wiki'
#we_model = 'wiki+hr'
we_model = 'hr'
#we_model = 'icaad'
#we_model = 'hr_stem+unstem'

removeNumbers = 0

THRESHOLD = 0.600
rm_stopwords = 0

SLIDING_WINDOW = 1
WINDOW_SIZE = 1

WEIGHTED = 1
STEMMING = 0
ps = PorterStemmer()

CORRECT_FOR_SENTENCE_LENGTH = 0


def cosine(u, v):
    return float(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))


def preprocess(sentence):
    #result = sentence.replace(",", " , ")
    #result = result.replace(":", " : ")
    #result = result.replace(";", " ; ")
    return re.sub('[^a-zA-Z0-9 ]+', ' ', sentence).lower()


def removeNonAlphabeticalChars(sentence):
    sentence = re.sub('[^a-zA-Z ]+', '', sentence)
    return sentence.strip()


def removeStopwords(sentence):
    tokens = sentence.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)


def transformProbability(p, a):
    return p**a/(p**a + (1-p)**a)


def avg_word_vectors(model, sentence, evidence, weighted=False):
    tokens = sentence.split()
    vec = [model.get_word_vector(word) for word in tokens]
    weights = [1] * len(tokens)
    if weighted:
        weights = []
        for word in tokens:
            if word in evidence.split():
                weights.append(2)
            elif word in EXCLUDE:
                weights.append(0.5)
            else:
                weights.append(1)
    return np.average(np.array(vec), weights=weights, axis=0)


def compute_similarity(evidence_embedding, sentence_embedding):
    similarity = []
    for sent_emb in sentence_embedding:
        similarity.append(cosine(evidence_embedding, sent_emb))
    return similarity


def sliding_window(iterable, size=WINDOW_SIZE):
    i = iter(iterable)
    window = []
    for elem in range(0, size):
        window.append(next(i))
    yield window
    for elem in i:
        window = window[1:] + [elem]
        yield window


def ICAAD_classification_use(data, category, config, evidences):

    print('** EMBEDDING **')
    t0 = time.time()

    data.drop_duplicates(subset='sentence', inplace=True)
    test_sentences = data.sentence.tolist()
    print(len(data))

    data['targetLabel'] = data.category == category

    processed_sentences = [preprocess(sentence) for sentence in test_sentences]
    if removeNumbers:
        processed_sentences = [removeNonAlphabeticalChars(sentence) for sentence in test_sentences]
    if STEMMING:
        stemmed_sentences = [[ps.stem(word) for word in sentence.split()] for sentence in processed_sentences]
        processed_sentences = [' '.join(tokens) for tokens in stemmed_sentences]
    data['proc_sentence'] = processed_sentences

    data = data[~(data['proc_sentence']=='')]
    test_sentences = data.proc_sentence.tolist()

    if ENCODING == 'USE':

        sent_encoder_graph = tf.get_default_graph()
        sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

        sentences = tf.placeholder(dtype=tf.string, shape=[None])
        embedding = sentenceEncoder(sentences)

        with sent_encoder_graph.as_default():
            with tf.Session(graph=sent_encoder_graph) as session:

                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                evidence_embedding = session.run(embedding, feed_dict={sentences: evidences})

                if rm_stopwords:
                    test_sentences = [removeStopwords(sentence) for sentence in test_sentences]
                    data['sentence'] = test_sentences
                sentence_embedding = session.run(embedding, feed_dict={sentences: test_sentences})

                session.close()

    elif ENCODING == 'fasttext':

        if we_model == 'wiki_40000':
            W2V_PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.bin'
        else:
            W2V_PATH = '../fastText/{}.bin'.format(we_model)

        model = fastText.load_model(W2V_PATH)
        print('VOCAB LENGTH: {}'.format(len(model.get_words())))

        evidences = [preprocess(evidence) for evidence in evidences]
        evidence_embedding = [avg_word_vectors(model, evidence, evidence, WEIGHTED) for evidence in evidences]

        if SLIDING_WINDOW:

            similarity = []
            for sentence in test_sentences:
                sent_embedding = []
                sentence_slides = sliding_window(sentence.split())
                for slide in sentence_slides:
                    sent_embedding.append(avg_word_vectors(model, ' '.join(slide), evidence, 0))
                sim = compute_similarity(evidence_embedding, sent_embedding)
                if len(sim) >= 1:
                    similarity.append(max(sim))
                else:
                    similarity.append(0.0)

        else:
            sentence_embedding = [avg_word_vectors(model, sentence, evidence, WEIGHTED) for sentence in test_sentences]
            similarity = compute_similarity(evidence_embedding, sentence_embedding)

    if CORRECT_FOR_SENTENCE_LENGTH:
        diffs = [abs(len(sentence.split())-len(evidences[0].split())) for sentence in test_sentences]
        extra = [diff*0.01 for diff in diffs]
        similarity = [sim + extra[ind] for ind, sim in enumerate(similarity)]

    data['similarity'] = similarity
    data['predictedLabel'] = data.similarity >= THRESHOLD

    computation_time = time.time() - t0

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

    html_name = config + '_' + ENCODING + '_' + we_model
    if WEIGHTED:
        html_name = html_name + '_weighted'
    if STEMMING:
        html_name = html_name + '_stem'
    if CORRECT_FOR_SENTENCE_LENGTH:
        html_name = html_name + '_sentLength'
    if SLIDING_WINDOW:
        html_name = html_name + '_window'

    data.to_pickle(html_name + '.pkl')

    axes = data.hist('similarity')
    plt.savefig(html_name + '_prediction.png')

    axes = data.hist('similarity', by='tags')
    plt.savefig(html_name + '_tags.png')

    data.sort_values(['tags', 'similarity'], inplace=True, ascending=[True, False])
    pd.set_option('display.max_colwidth', 500)

    viewer = Viewer('use')
    viewer.use_classificationResults(html_name, evidences, data.drop(columns=['id', 'category', 'predictedLabel', 'targetLabel']), ENCODING, we_model, THRESHOLD, evaluation, computation_time)


if __name__ == '__main__':

    data = pd.read_csv(PATH, encoding='utf8')

    category = 'Evidence.of.SA'
    negCategory = 'Evidence.no.SADV'
    data = data[(data.category == category) | (data.category == negCategory)]

    set_evidences = [('phrase_rape', ['rape is committed when having sexual intercourse without consent'])]
    set_evidences = [('phrase_rape', ['rape sexual assault abuse carnal knowledge'])]
    set_evidences = [('keyword_rape', ['rape'])]

    for config, evidences in set_evidences:
        ICAAD_classification_use(data, category, config, evidences)
