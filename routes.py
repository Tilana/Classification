from flask import Flask, request
import lda.osHelper as osHelper
from tensorflow import errors as tensorflowErrors
from train import train;
from predictDoc import predictDoc;
from shutil import rmtree
import pandas as pd
import json
import os
import tensorflow as tf
import argparse as _argparse
# import pdb

app = Flask(__name__)

@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)

    sentences = pd.read_json('[' + json.dumps(data) + ']');
    sentences['sentence'] = sentences['evidence'][0]['text']
    sentences['label'] = sentences['isEvidence'][0]
    # pdb.set_trace()

    train(sentences, data['value'] + data['property'])

    return "{}"

@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    data = json.loads(request.data)
    property = data['property']
    value = data['value']
    evidences = pd.read_json(json.dumps(data['evidences']));

    rmtree(os.path.join('runs', value+property), ignore_errors=True)
    tf.app.flags._global_parser = _argparse.ArgumentParser()

    train(evidences, value + property)

    return "{}"

@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']));

    property = data['property'];
    value = data['value'];

    model = value+property;

    results = [];
    for doc in docs.iterrows():
        try:
            predictions = predictDoc(doc[1], model);
            predictions = predictions.rename(index=str, columns={'text': 'evidence'});
            predictions['property'] = property;
            predictions['value'] = value;
            predictions['document'] = doc[1]['_id']
            results.append(predictions);
        except:
            print 'model not trained'

    return pd.concat(results).to_json(orient='records')

@app.route('/classification/predict', methods=['POST'])
def predict_route():
    data = json.loads(request.data)
    evidencesData = data['properties']
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']').loc[0];
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

    return pd.concat(results).to_json(orient='records')
