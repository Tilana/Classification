# export TFHUB_CACHE_DIR=./tfhub_modules
# export FLASK_APP=routes.py
# flask run --port 4000

from flask import Flask, request
import lda.osHelper as osHelper
from tensorflow import errors as tensorflowErrors
from sentenceTokenizer import tokenize
import tensorflow_hub as hub
from train import train;
from predictDoc import predictDoc;
from shutil import rmtree
import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
import argparse as _argparse
import pdb
import subprocess
from systemd import journal

app = Flask(__name__)

THRESHOLD = 0.67
MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 5
MIN_NUM_TRAINING_SENTENCES = 20

TRAINING_FILE = 'training.csv'

#subprocess.Popen('python lda/WordEmbedding.py', shell=True)

def get_similar_sentences(similarity, evidences, sentences, doc_id):
    similar_sentences = pd.DataFrame(columns=['probability'])
    for ind, sentence in enumerate(sentences):
        for pos,sim in enumerate(similarity[:,ind]):
            if sim>=THRESHOLD:
                similar_sentences = similar_sentences.append({'evidence':sentence, 'probability':sim, 'label':1, 'document':doc_id, 'property':evidences.loc[pos]['property'], 'value':evidences.loc[pos]['value']}, ignore_index=True)
    return similar_sentences


@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)

    label = data['property'] + "_" + data['value'] + "_" + str(data['isEvidence']);
    sentence = data['evidence']['text'].encode('utf-8');

    df = pd.DataFrame({'sentence': sentence, 'property': data['property'], 'value':data['value'], 'label': data['isEvidence']}, index=[0])
    if os.path.exists(TRAINING_FILE):
        df.to_csv(TRAINING_FILE, mode='a', header=False, index=False, encoding='utf8')
    else:
        df.to_csv(TRAINING_FILE, mode='a', index=False, encoding='utf8')
    return "{}"

@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    data = json.loads(request.data)
    model = data['value'] + data['property']

    #model_evidences = pd.read_json(json.dumps(data['evidences']), encoding='utf8');

    evidences = pd.read_csv(TRAINING_FILE, encoding='utf8')
    model_evidences = evidences[(evidences['property']==data['property']) & (evidences['value']==data['value'])]

    if len(model_evidences) >= MIN_NUM_TRAINING_SENTENCES:
        journal.send('CNN TRAINING')
        rmtree(os.path.join('runs', model), ignore_errors=True)
        tf.app.flags._global_parser = _argparse.ArgumentParser()
        train(model_evidences, model)
    elif len(model_evidences) == 0:
        journal.send('NO TRAINING DATA IS AVAILABLE')
    else:
        journal.send('NOT ENOUGH DATA FOR CNN TRAINING')

    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():

    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf8');


    evidences = pd.read_csv(TRAINING_FILE, encoding='utf8')
    model_evidences = evidences[(evidences['property']==data['property']) & (evidences['value']==data['value'])]
    model_evidences = model_evidences.reset_index()


    if len(model_evidences)==0:
        return "{}"

    elif len(model_evidences)< MIN_NUM_TRAINING_SENTENCES:
        journal.send('UNIVERSAL SENTENCE ENCODER')
        tf.reset_default_graph()
        sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        journal.send('LOADED SENTENCE ENCODER')

        suggestions = pd.DataFrame(columns=['probability'])

        sentences = tf.placeholder(dtype=tf.string, shape=[None])
        embedding = sentenceEncoder(sentences)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            journal.send('ENCODE EVIDENCE SENTENCES')
            evidence_embedding = session.run(embedding, feed_dict={sentences: model_evidences.sentence.tolist()})

            docIDs = []

            for doc in docs.iterrows():
                docID = doc[1]['_id']

                if docID not in docIDs:
                    journal.send('ENCODE DOC ' + docID)
                    doc_sentences = tokenize(doc[1].text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
                    sentence_embedding = session.run(embedding, feed_dict={sentences: doc_sentences})

                    similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
                    suggestions = suggestions.append(get_similar_sentences(similarity, model_evidences, doc_sentences, docID))
                    docIDs.append(docID)

            session.close()

        if len(suggestions)>0:
            suggestions.sort_values(by=['probability'], ascending=False, inplace=True)
            suggestions.drop_duplicates(suggestions.columns.difference(['probability']), inplace=True)
            suggestions = suggestions[~suggestions.evidence.isin(model_evidences.sentence)]
            suggestions.reset_index(inplace=True)
        return suggestions.to_json(orient='records')

    else:
        journal.send('CONVOLUTIONAL NEURAL NET')
        model = data['value']+data['property'];
        results = [];
        for doc in docs.iterrows():
            try:
                predictions = predictDoc(doc[1], model);
                predictions = predictions.rename(index=str, columns={'sentence': 'evidence', 'predictedLabel':'label'});
                predictions['property'] = data['property'];
                predictions['value'] = data['value'];
                predictions['document'] = doc[1]['_id']
                results.append(predictions)
            except:
                journal.send('model not trained')
        suggestions = pd.concat(results).sort_values(by=['probability'], ascending=False).head(100)
        return suggestions.to_json(orient='records')



@app.route('/classification/predict', methods=['POST'])
def predict_route():

    data = json.loads(request.data)
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']', encoding='utf8').loc[0]

    evidenceData = pd.DataFrame.from_dict(data['properties'])
    evidenceData['prop+value'] = evidenceData['property'] + '_' + evidenceData['value']
    categories = evidenceData['prop+value'].unique()

    evidences = pd.read_csv(TRAINING_FILE, encoding='utf8')
    evidences['prop+value'] = evidences['property'] + '_' + evidences['value']

    evidences = evidences[evidences['prop+value'].isin(categories)]
    evidences.reset_index(inplace=True)

    tf.reset_default_graph()
    sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    suggestions = pd.DataFrame(columns=['probability'])

    sentences = tf.placeholder(dtype=tf.string, shape=[None])
    embedding = sentenceEncoder(sentences)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        doc_sentences = tokenize(doc.text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
        sentence_embedding = session.run(embedding, feed_dict={sentences: doc_sentences})
        evidence_embedding = session.run(embedding, feed_dict={sentences: evidences.sentence.tolist()})

        similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
        suggestions = suggestions.append(get_similar_sentences(similarity, evidences, doc_sentences, evidenceData.document[0]))

        session.close()

    if len(suggestions)>0:
        suggestions.sort_values(by=['probability'], ascending=False, inplace=True)
        suggestions.drop_duplicates(suggestions.columns.difference(['probability']), inplace=True)
        suggestions = suggestions[~suggestions.evidence.isin(evidences.sentence)]
        suggestions.reset_index(inplace=True)
    return suggestions.to_json(orient='records')

