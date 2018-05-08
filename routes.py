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
app = Flask(__name__)

THRESHOLD = 0.67
MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 5

TRAINING_FILE = 'training.csv'

sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")


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

    rmtree(os.path.join('runs', model), ignore_errors=True)
    tf.app.flags._global_parser = _argparse.ArgumentParser()

    train(model_evidences, model)

    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf8');

    evidences = pd.read_csv(TRAINING_FILE, encoding='utf8')
    model_evidences = evidences[(evidences['property']==data['property']) & (evidences['value']==data['value']) & (evidences['label']==True)]
    model_evidences = model_evidences.reset_index()

    if len(model_evidences)==0:
        return "{}"

    if len(model_evidences)<10:
        suggestions = pd.DataFrame(columns=['probability'])
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            evidence_embedding = session.run(sentenceEncoder(model_evidences.sentence.tolist()))

            for doc in docs.iterrows():
                sentences = tokenize(doc[1].text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
                sentence_embedding = session.run(sentenceEncoder(sentences))

                similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
                suggestions = suggestions.append(get_similar_sentences(similarity, model_evidences, sentences, doc[1]['_id']))
                suggestions.sort_values(by=['probability'], ascending=False, inplace=True)
                suggestions.drop_duplicates(inplace=True)

            session.close()
        return suggestions.to_json(orient='records')

    else:
        model = data['value']+data['property'];
        results = [];
        for doc in docs.iterrows():
            try:
                predictions = predictDoc(doc[1], model);
                predictions = predictions.rename(index=str, columns={'text': 'evidence'});
                predictions['property'] = data['property'];
                predictions['value'] = data['value'];
                predictions['document'] = doc[1]['_id']
                results.append(predictions);
            except:
                print 'model not trained'
        return pd.concat(results).sort_values(by=['probability'], ascending=False).head(100).to_json(orient='records')



@app.route('/classification/predict', methods=['POST'])
def predict_route():
    data = json.loads(request.data)
    evidencesData = data['properties']
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']', encoding='utf8').loc[0];
    results = [];
    for evidence in evidencesData:
        try:
            predictions = predictDoc(doc, evidence['value'] + evidence['property']);
            predictions = predictions.rename(index=str, columns={'text': 'evidence'});
            predictions['property'] = evidence['property'];
            predictions['document'] = evidence['document'];
            predictions['value'] = evidence['value'];
            results.append(predictions);
        except:
            print 'model not trained'

    return pd.concat(results).sort_values(by=['probability'], ascending=False).head(100).to_json(orient='records')
