#import sys
#sys.path.insert(0, '/home/natalie/Documents/Huridocs-Code/Classification/lda')
from fastai.text import *
from lda.Evaluation import Evaluation
from lda.Viewer import Viewer
from lda.ImagePlotter import ImagePlotter
import pandas as pd
import numpy as np
import time
import pdb

DATA_PATH = '../data/'
filename = 'sentences_ICAAD.csv'

train = 0
EVAL = 0

WIKI_MODEL = 'wt103/fwd_wt103'
WIKI_VOCAB = 'wt103/itos_wt103'

SENTENCES = ['rape', 'raped', 'incest', 'murder', 'assault', 'apple', 'court', 'the girl was raped', 'with 2 counts of rape', 'he sexually assaulted her', 'forcefully inserted his penis', 'the final sentence is imprisonment', 'crimes of murder and manslaughter', 'assault occaising actual bodily harm', 'robbery with violence']


def cosine(u, v):
    return float(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))


def encode(model, text, avg=1):
    model.model.reset()
    vocab, _ = model.data.one_item(text)
    tensor = vocab.clone().detach()
    enc = model.model(tensor)[-1][-1]
    if avg:
        return enc[0].mean(0).tolist()
    else:
        return enc[0].max(0)[0].tolist()

t0 = time.time()

ICAAD_data = pd.read_csv(DATA_PATH + filename, index_col=0)
ICAAD_data = ICAAD_data[ICAAD_data['category'].isin(['Evidence.no.SADV', 'Evidence.of.SA'])]
train_df = ICAAD_data.sample(50, random_state=42)
valid_df = ICAAD_data.drop(train_df.index).sample(20, random_state=42)
test_df = ICAAD_data.drop(train_df.index.append(valid_df.index))

data_lm = TextLMDataBunch.from_df('../data/', train_df=train_df, valid_df=valid_df, bs=32, text_cols='text', label_cols='category')
data_clas = TextClasDataBunch.from_df(path=DATA_PATH, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols='text', label_cols='category', vocab=data_lm.train_ds.vocab, bs=32)

learn = language_model_learner(data_lm, pretrained_fnames=[WIKI_MODEL, WIKI_VOCAB], drop_mult=0.5)

if train:
    print('*** Finetune Language Model ***')
    learn.freeze()
    learn.fit_one_cycle(1, 1e-3, div_factor=20, wd=1e-7)
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3, div_factor=20, wd=1e-7)

    learn.save_encoder('ICAAD_sentences')

if EVAL:
    learn.load_encoder('ICAAD_sentences')
    sim = []
    for sent1 in SENTENCES:
        curr_sim = []
        enc1 = encode(learn, sent1, avg=0)
        for sent2 in SENTENCES:
            enc2 = encode(learn, sent2)
            curr_sim.append(cosine(enc1, enc2))
        sim.append(curr_sim)

    plotter = ImagePlotter(1)
    plotter.heatmap(np.array(sim), './ulmfit/sentences.jpg', SENTENCES, SENTENCES)


text_classifier = text_classifier_learner(data_clas, drop_mult=0.5)
text_classifier.load_encoder('ICAAD_sentences')

print('*** Train Classifier ***')
text_classifier.fit_one_cycle(4, 1e-2)
text_classifier.freeze_to(-2)
text_classifier.fit_one_cycle(1, slice(5e-3/2., 5e-3))
text_classifier.unfreeze()
text_classifier.fit_one_cycle(1, slice(2e-3/100, 2e-3))

computation_time = time.time() - t0

print('*** Evauation ***')
predictions = text_classifier.get_preds(DatasetType.Test, ordered=True)[0]
values, indices = predictions.max(1)
test_df['probability'] = values

mapping = {'Evidence.no.SADV':0, 'Evidence.of.SA':1}
targetLabel = test_df.replace({'category': mapping}).category

evaluation = Evaluation(targetLabel.tolist(), prediction=indices.tolist())
evaluation.computeMeasures()
evaluation.confusionMatrix()

print('Accuracy: ' + str(evaluation.accuracy))
print('Recall: ' + str(evaluation.recall))
print('Precision: ' + str(evaluation.precision))
print(evaluation.confusionMatrix)

evaluation.createTags()
test_df['tags'] = evaluation.tags
test_df.sort_values(['tags', 'probability'], inplace=True, ascending=[True, False])

viewer = Viewer('ulmfit')
info = {'name': 'ulmfit', 'Nr train data': len(train_df), 'Nr valid data': len(valid_df), 'Nr test data': len(test_df), 'Computation Time': computation_time}
viewer.classification(info, test_df.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'docID', 'label']), evaluation)
