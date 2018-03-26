#! /usr/bin/env python
import tensorflow as tf
import webbrowser
from lda import NeuralNet, Preprocessor, Info, ImagePlotter
import numpy as np
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from lda.osHelper import generateModelDirectory
import pdb

DISPLAY_THRESHOLD = 30
wordFrequencies = ['POS_WORD_FREQUENCY', 'NEG_WORD_FREQUENCY']

def wordRelevance(sentence, category):

    model_path = generateModelDirectory(category)

    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')
    infoFile = os.path.join(model_path, 'info.json')

    debugFile = os.path.join(model_path, 'debug.html')
    f = open(debugFile, 'w')
    f.write('<html><head><h1>DEBUG MODE</h1></head><body>')
    f.write('<p>Category:  %s </p>' % category)
    f.write('<p>Sentence:  %s </p>' % sentence)

    info = Info(infoFile)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)
    vocabulary = vocab_processor.vocabulary_

    plotter = ImagePlotter(False)

    if info.preprocessing:
        f.write('<h4><br>PREPROCESSING</br></h4>')
        preprocessor = Preprocessor()
        sentence = preprocessor.cleanText(sentence)
        f.write('<p>%s</p>' % sentence)

    #sentences = sent_tokenize(doc.text)
    #sentenceDB = pd.DataFrame(sentences, columns=['text'])

    #sentenceDB['sentenceLength'] = sentenceDB.text.map(setSentenceLength)
    #sentenceDB = sentenceDB[sentenceDB.sentenceLength.map(filterSentenceLength)]
    #sentenceDB['text'] = sentenceDB['text'].str.lower()

    f.write('<h4><br>VOCAB MAPPING </br></h4>')

    occurences = pd.DataFrame(columns=['word', 'POS', 'NEG'])

    for word in sentence.split():
        f.write('<tr><td>%s: </td> <td> %d </td></tr>' % (word, vocabulary.get(word)))
        occurences = occurences.append({'word':word, 'POS':info.POS_WORD_FREQUENCY.get(word), 'NEG':info.NEG_WORD_FREQUENCY.get(word)}, ignore_index=True)
    occurences.fillna(0, inplace=True)
    occurences.set_index('word', inplace=True)

    plt.figure()
    occurences.plot(kind='bar', title='Word Occurence In Training Data', color=['g', 'r'])
    plt.savefig(model_path+'/'+'WordOccurence.jpg')
    plt.close()


    X_val = np.array(list(vocab_processor.transform([sentence])))

    f.write('<p> Model Input:  %s</p>' % X_val[0])

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:

            nn.loadCheckpoint(graph, sess, checkpoint_dir)


            f.write('<h4><br>RESULTS</br></h4>')
            validationData = {nn.X: np.asarray(X_val), nn.pkeep:1.0}
            prediction, probability = sess.run([nn.predictions, nn.probability], feed_dict=validationData)


            f.write('<p> Prediction:  %s </p>' % prediction[0])
            f.write('<p> Probability: %s </p>' % probability[0])

            wordIDs = X_val[0]
            sentenceLength = len(wordIDs)
            probabilities = [0] * sentenceLength

            for blankPos in range(sentenceLength):
                if wordIDs[blankPos] != 0:
                    blankedInput = wordIDs
                    blankedInput[blankPos] = 0

                    wordRelevanceData = {nn.X: np.asarray([blankedInput]), nn.pkeep:1.0}
                    probability = sess.run(nn.probability, feed_dict=wordRelevanceData)
                    probabilities[blankPos] = probability[0]

            print probabilities

            plotter.heatmap(pd.DataFrame(probabilities), model_path+'/wordRelevance.jpg')

            sess.close()


    f.write('<h4><br>Word Relevance<br></h4><table>')
    f.write('<img src="%s" alt="wrong path" height="580">' % ('wordRelevance.jpg'))

    f.write('<h4><br>MODEL ANALYSIS</br></h4><table>')
    infoFeatures = ['TOTAL_NR_TRAIN_SENTENCES', 'NR_TRAIN_SENTENCES_POS', 'NR_TRAIN_SENTENCES_NEG']
    for infoFeature in infoFeatures:
        f.write('<tr><td>%s: </td> <td> %d </td></tr>' % (infoFeature, getattr(info, infoFeature)))

    f.write('<tr><td>%s</td> <td> %s </td></tr>' % ('', ''))
    f.write('<tr><td>%s</td> <td> %s </td></tr>' % ('', ''))

    for wordFrequency in wordFrequencies:
        f.write('<tr><td>%s: </td> <td> %d </td></tr>' % (wordFrequency, len(getattr(info,wordFrequency))))


    f.write('<tr><td>DISPLAY_THRESHOLD: </td> <td> %d </td></tr>' % DISPLAY_THRESHOLD)
    f.write('</table>')
    for wordFrequency in wordFrequencies:
        plotPath= model_path + '/' + wordFrequency + '.jpg'
        frequency = getattr(info, wordFrequency)
        frequency = [(v, k) for v,k in frequency.iteritems()]
        frequency = sorted(frequency, key=lambda x:(-x[1], x[0]))
        frequency = frequency[:DISPLAY_THRESHOLD]
        frequency = dict(frequency)
        plotter.barplot(frequency.values(), ylabel=frequency.keys(), log=False, title=wordFrequency, path=plotPath)
        f.write('<img src="%s" alt="wrong path" height="580">' % (wordFrequency+'.jpg'))


    f.write('<img src="%s" alt="wrong path" height="580">' % ('WordOccurence.jpg'))
    f.write('<p> OOV:  %s</p>' % info.OOV)

    f.write('</body></html>')
    f.close()
    webbrowser.open_new_tab(debugFile)



if __name__=='__main__':
    #wordRelevance('This is a test sentence with domestic violence in ITS name', 'ICAAD_DV_sentences')
    wordRelevance('the accused has been charged with assault occasioning actual boily harm contrary to section 275 of the crimes decree no 4', 'ICAAD_DV_sentences')
    #wordRelevance('He beats his wife cause grievous pain.', 'ICAAD_DV_sentences')
    #wordRelevance('The incedent was a misunderstanding and could be resolved without further problems', 'ICAAD_DV_sentences')

