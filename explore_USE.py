import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from math import ceil
import re
import pdb

stopwords = ['of', 'the', 'for', 'an', 'in', 'and', 'to', 'a', 'or', 'that', 'as']

sent_encoder_graph = tf.get_default_graph()
sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

sentences = tf.placeholder(dtype=tf.string, shape=[None])
embedding = sentenceEncoder(sentences)


def removeStopwords(sentence):
    tokens = [word for word in sentence.split() if word not in stopwords]
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


def sentenceVariations(sentence):
    variations = []
    variations_label = []
    words = sentence.split()
    for ind in range(len(words)):
        variation = words[:ind] + words[ind+1:]
        variations.append(' '.join(variation))
        variations_label.append('rm_' + words[ind])
    return (variations, variations_label)


def explore_USE(evidence, sentence):

    sentence_removedStopwords = removeStopwords(sentence)
    sentence_letters = removeNonLettres(sentence)
    sentence_rmSpecialChars = removeSpecialChars(sentence)
    sentence_lower = sentence.lower()

    (variations, variations_label) = sentenceVariations(sentence)
    (first_half, second_half) = splitSentenceInHalf(sentence)
    sentence_variations = [sentence, sentence_removedStopwords, sentence_letters, sentence_rmSpecialChars, sentence_lower, first_half, second_half]
    sentence_variations.extend(variations)

    label = ['original', 'rm_stopwords', 'rm_letters', 'rm_specialChars', 'lowerCase', 'first_half', 'second_half']
    label.extend (variations_label)

    with sent_encoder_graph.as_default():
        with tf.Session(graph=sent_encoder_graph) as session:

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            evidence_embedding = session.run(embedding, feed_dict={sentences: evidence})

            print sentence_variations

            sentence_embedding = session.run(embedding, feed_dict={sentences: sentence_variations})

            similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
            similarity = similarity.squeeze().tolist()

            results = zip(similarity, sentence_variations, label)
            results = sorted(results, reverse=True)

        session.close()


    return results


if __name__=="__main__":
    all_results = []
    evidences = [['rape'], ['rape is committed when having sexual intercourse without consent']]
    evidences = [['Islamic headscarf']]
    #evidences = [['Islamic headscarf'], ['prohibit the wearing of the Islamic headscarf']]

    for evidence in evidences:
        print evidence
        results = []
        SENTENCES = ['It observes in this connection that, as a Muslim woman who for religious reasons wishes to wear the full - face veil in public,', 'his private or family life (see, mutatis mut andis, Dahlab v. Switzerland (dec.), no. 42393/98, ECHR 2001 - V (requirement for a teacher not to wear a headscarf),', 'the Government added that the precepts of Islam should be taken into consideration in determining the place occupied by the Alevi faith within the Muslim religion', 'had made a statement that " Muslim were not a people, that they did not possess culture and that, accordingly, destroying mosques could not be seen as a destruction of cultural monuments']
        SENTENCES = ['the starting point for the rape of an adult in fiji is 7 years imprisonment', 'rape contrary to sections 207 (1) and (2) (a) and (3) of the crimes decree no', 'he removed her clothes and his own, and he raped her there underneath that house', '[10] this is a case of incestrial rape, the tariff varies from 10 years to 16 years']
        SENTENCES = ['Islamic', 'headscarf', 'religious']
        SENTENCES= ['Muslim', 'full-face veil', 'apple']
        #pdb.set_trace()
        #sentences = ['his private or family life (see, mutatis mut andis, Dahlab v. Switzerland (dec.), no. 42393/98, ECHR 2001 - V (requirement for a teacher not to wear a headscarf),'

        #pdb.set_trace()
        #stopwordSetting = [False, True]
        #for proc_setting in removeStopwords:
        for sentence in SENTENCES:
            curr_results = explore_USE(evidence, sentence)
            curr_results = pd.DataFrame(curr_results, columns=['similarity', str(evidence), 'label'])
            curr_results.drop_duplicates(inplace=True)
            org_score = curr_results[curr_results['label']=='original'].similarity.tolist()[0]
            curr_results['diff'] = curr_results['similarity'] - org_score
            curr_results.sort_values('similarity', inplace=True, ascending=False)
            curr_results.reset_index(inplace=True, drop=True)
            results.append(curr_results)

        results = pd.concat(results, axis=0)
        all_results.append(results)

        #pdb.set_trace()

        #results = sum(results, [])
        #results = pd.DataFrame(results, columns=['similarity', str(evidence), 'label'])

        #results.drop_duplicates(inplace=True)
        #results.sort_values('similarity', inplace=True, ascending=False)
        #all_results.append(results)

    all_res = pd.concat(all_results, axis=1)
    #all_res.to_csv('use_explore_headscarf.csv', index=False)
    all_res.to_csv('use_explore_rape.csv', index=False)

