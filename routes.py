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
from lda import NeuralNet
import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
import argparse as _argparse
import pdb
import subprocess
from systemd import journal
import pymongo
from bson.json_util import dumps

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.machine_learning
mongo_suggestions = db.suggestions
mongo_training = db.training

app = Flask(__name__)

THRESHOLD = 0.67
MAX_SENTENCE_LENGTH = 40
MIN_SENTENCE_LENGTH = 5
MIN_NUM_TRAINING_SENTENCES = 20

#subprocess.Popen('python lda/WordEmbedding.py', shell=True)

def get_similar_sentences(similarity, evidences, sentences, doc_id):
    similar_sentences = pd.DataFrame(columns=['probability'])
    for ind, sentence in enumerate(sentences):
        for pos,sim in enumerate(similarity[:,ind]):
            if sim >= THRESHOLD:
                mongo_suggestions.insert_one({'evidence':sentence, 'probability':str(sim), 'label':1, 'document':doc_id, 'property':evidences.loc[pos]['property'], 'value':evidences.loc[pos]['value']})

@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)

    sentence = data['evidence']['text'].encode('utf-8');
    label = str(data['isEvidence'])
    mongo_training.insert_one({'property': data['property'], 'value': data['value'], 'sentence': sentence, 'label': label})

    return "{}"

@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    data = json.loads(request.data)
    model = data['value'] + data['property']

    evidences = mongo_training.find({'property': data['property'], 'value':  data['value']})
    evidences = pd.DataFrame(list(evidences))
    posEvidences = mongo_training.find({'property': data['property'], 'value':  data['value'], 'label':'True'})
    nrPosEvidences = len(list(posEvidences))

    if nrPosEvidences >= MIN_NUM_TRAINING_SENTENCES:
        journal.send('CNN TRAINING')
        rmtree(os.path.join('runs', model), ignore_errors=True)
        tf.app.flags._global_parser = _argparse.ArgumentParser()
        train(evidences, model)
    elif nrPosEvidences == 0:
        journal.send('NO TRAINING DATA IS AVAILABLE')
    else:
        journal.send('NOT ENOUGH DATA FOR CNN TRAINING')

    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():

    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf8');

    evidences = mongo_training.find({'property': data['property'], 'value':  data['value'], 'label':'True'})
    evidences = pd.DataFrame(list(evidences))

    if len(evidences)==0:
        return "{}"

    elif len(evidences) < MIN_NUM_TRAINING_SENTENCES:
        journal.send('UNIVERSAL SENTENCE ENCODER')
        tf.reset_default_graph()
        sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        journal.send('LOADED SENTENCE ENCODER')

        mongo_suggestions.remove()

        sentences = tf.placeholder(dtype=tf.string, shape=[None])
        embedding = sentenceEncoder(sentences)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            journal.send('ENCODE EVIDENCE SENTENCES')
            evidence_embedding = session.run(embedding, feed_dict={sentences: evidences.sentence.tolist()})

            docIDs = []

            for doc in docs.iterrows():
                docID = doc[1]['_id']

                if docID not in docIDs:
                    journal.send('ENCODE DOC ' + docID)
                    doc_sentences = tokenize(doc[1].text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)
                    sentence_embedding = session.run(embedding, feed_dict={sentences: doc_sentences})

                    similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
                    get_similar_sentences(similarity, evidences, doc_sentences, docID)
                    docIDs.append(docID)

            session.close()

        result = dumps(mongo_suggestions.find({},{'_id':0}).limit(10).sort("probability", -1))
        return result

    else:
        journal.send('CONVOLUTIONAL NEURAL NET')
        model = data['value']+data['property']
        results = []
        docIDs = []

        nn = NeuralNet()
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as session:

                model_path = osHelper.generateModelDirectory(model)
                checkpoint_dir = os.path.join(model_path, 'checkpoints')
                nn.loadCheckpoint(graph, session, checkpoint_dir)

                for doc in docs.iterrows():
                    docID = doc[1]['_id']

                    if docID not in docIDs:
                        journal.send('ENCODE DOC ' + docID)
                        predictions = predictDoc(doc[1], model, nn, session);
                        predictions = predictions.rename(index=str, columns={'sentence': 'evidence', 'predictedLabel':'label'});
                        predictions['property'] = data['property'];
                        predictions['value'] = data['value'];
                        predictions['document'] = docID
                        results.append(predictions)
                        docIDs.append(docID)
                session.close()

        suggestions = pd.concat(results).sort_values(by=['probability'], ascending=False).head(100)
        return suggestions.to_json(orient='records')



@app.route('/classification/predict', methods=['POST'])
def predict_route():

    data = json.loads(request.data)
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']', encoding='utf8').loc[0]

    evidenceData = pd.DataFrame.from_dict(data['properties'])
    evidences = mongo_training.find({'property': {"$in": evidenceData.property.tolist()}, 'value': {"$in": evidenceData.value.tolist()}, 'label':'True'})
    evidences = pd.DataFrame(list(evidences))

    if len(evidences)==0:
        journal.send('ERROR: No training data is available')
        return "{}"

    tf.reset_default_graph()
    sentenceEncoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
    sentences = tf.placeholder(dtype=tf.string, shape=[None])
    embedding = sentenceEncoder(sentences)

    suggestions = pd.DataFrame(columns=['probability'])
    doc_sentences = tokenize(doc.text, MAX_SENTENCE_LENGTH, MIN_SENTENCE_LENGTH)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        sentence_embedding = session.run(embedding, feed_dict={sentences: doc_sentences})
        evidence_embedding = session.run(embedding, feed_dict={sentences: evidences.sentence.tolist()})

    session.close()

    similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
    get_similar_sentences(similarity, evidences, doc_sentences, evidenceData.document[0])
    result = dumps(mongo_suggestions.find({},{'_id':0}).limit(10).sort("probability", -1))
    return result

