import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from math import ceil
import re
import fastText
import pdb

STOPWORDS = ['of', 'the', 'for', 'an', 'at', 'with', 'when', 'her', 'in', 'and', 'to', 'a', 'or', 'that', 'as']

WEIGHTED = False

we_model = 'wiki'
we_model = 'hr'


def removeStopwords(sentence):
    tokens = [word for word in sentence.split() if word not in STOPWORDS]
    return ' '.join(tokens)


def removeSpecialChars(sentence):
    return re.sub('[^A-Za-z0-9 ]+', '', sentence)


def removeNonLettres(sentence):
    return re.sub('[^A-Za-z ]+', '', sentence)


def splitSentenceInHalf(sentence):
    words = sentence.split()
    first_half = words[:len(words)/2]
    second_half = words[len(words)/2:]
    return [' '.join(first_half), ' '.join(second_half)]


def cosine(u, v):
    return float(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))


def compute_similarity(evidence_embedding, sentence_embedding):
    similarity = []
    for sent_emb in sentence_embedding:
        similarity.append(cosine(evidence_embedding, sent_emb))
    return similarity


def sentenceVariations(sentence):
    variations = []
    variations_label = []
    words = sentence.split()
    for ind in range(len(words)):
        variation = words[:ind] + words[ind+1:]
        variations.append(' '.join(variation))
        variations_label.append('rm_' + words[ind])
    return (variations, variations_label)


def avg_word_vectors(model, sentence, evidence, weighted=False):
    tokens = sentence.split()
    vec = [model.get_word_vector(word) for word in tokens]
    weights = [1] * len(tokens)
    if weighted:
        weights = []
        for word in tokens:
            if word in evidence.split():
                weights.append(1.5)
            elif word in STOPWORDS:
                weights.append(0.5)
            else:
                weights.append(1)
        # weights = [1.5 if word in evidence.split() else 1 for word in tokens]
    return np.average(np.array(vec), weights=weights, axis=0)


def explore_fasttext(evidence, sentence):

    sentence_removedStopwords = removeStopwords(sentence)
    sentence_letters = removeNonLettres(sentence)
    sentence_rmSpecialChars = removeSpecialChars(sentence)
    sentence_lower = sentence.lower()

    (variations, variations_label) = sentenceVariations(sentence)
    (first_half, second_half) = splitSentenceInHalf(sentence)
    sentence_variations = [sentence, sentence_removedStopwords, sentence_letters, sentence_rmSpecialChars, sentence_lower, first_half, second_half]
    sentence_variations.extend(variations)

    label = ['original', 'rm_stopwords', 'rm_numbers', 'rm_specialChars', 'lowerCase', 'first_half', 'second_half']
    label.extend(variations_label)
    label.append('weighted')


    if we_model == 'wiki_40000':
        W2V_PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.bin'
    elif '_incr' in we_model:
        W2V_PATH = '../fastText_incr/fastText/{}.bin'.format(we_model.split('_')[0])
    else:
        W2V_PATH = '../fastText/{}.bin'.format(we_model)

    model = fastText.load_model(W2V_PATH)

    sentence_embedding = [avg_word_vectors(model, sentence, evidence, WEIGHTED) for sentence in sentence_variations]
    weighted_sentence = avg_word_vectors(model, sentence, evidence, 1)
    sentence_embedding.append(weighted_sentence)
    evidence_embedding = avg_word_vectors(model, evidence, evidence, WEIGHTED)

    similarity = compute_similarity(evidence_embedding, sentence_embedding)

    sentence_variations.append(sentence)
    results = zip(similarity, sentence_variations, label)
    results = sorted(results, reverse=True)

    return results


if __name__ == "__main__":
    all_results = []
    evidences = ['rape', 'rape is committed when having sexual intercourse without consent']

    for evidence in evidences:
        print evidence
        results = []
        SENTENCES = ['rape contrary to sections 149 and 150 of the penal code cap',
                    '8 - rape - 12 years 10 months', 'he removed her lower clothing and raped her',
                    'the medical report confirmed the above sexual abuse',
                    'count 1 statement of offence robbery with violence:',
                    'the girl was put in your custody at the age of 3 years, when her parents separated.']

        for sentence in SENTENCES:
            curr_results = explore_fasttext(evidence, sentence)
            curr_results = pd.DataFrame(curr_results, columns=['similarity', str(evidence), 'label'])
            curr_results.drop_duplicates(inplace=True)
            org_score = curr_results[curr_results['label'] == 'original'].similarity.tolist()[0]
            curr_results['diff'] = curr_results['similarity'] - org_score
            curr_results.sort_values('similarity', inplace=True, ascending=False)
            curr_results.reset_index(inplace=True, drop=True)
            results.append(curr_results)

        results = pd.concat(results, axis=0)
        all_results.append(results)

    all_res = pd.concat(all_results, axis=1)
    all_res.to_csv('ft_explore_rape.csv', index=False)

