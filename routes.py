from flask import Flask, request
from tensorflow import errors as tensorflowErrors
from train import train;
from predictDoc import predictDoc;
import pandas as pd
import json

app = Flask(__name__)

@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)

    sentence = data['evidence']['text']

    train(sentence, data['value'] + data['property'], data['isEvidence'])

    return "{}"

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
