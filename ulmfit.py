import sys
sys.path.insert(0, '/home/natalie/Documents/Huridocs-Code/Classification/lda')
from fastai.text import *
from lda.Evaluation import Evaluation
from lda.Viewer import Viewer
import pandas as pd
import pdb

DATA_PATH = '../data/'
filename = 'sentences_ICAAD.csv'

train = 0

WIKI_MODEL = 'wt103/fwd_wt103'
WIKI_VOCAB = 'wt103/itos_wt103'

ICAAD_data = pd.read_pickle(DATA_PATH + 'ICAAD/ICAAD.pkl')
ICAAD_data = ICAAD_data[['title', 'text', 'Sexual.Assault.Manual']]


def cosine(u, v):
    return float(np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v)))


def encode(model, text):
    model.model.reset()
    vocab, _ = model.data.one_item(text)
    tensor = torch.tensor(vocab).clone().detach()
    enc = model.model(tensor)[-1][-1]
    avg_enc = enc[0].mean(0).tolist()
    max_enc = enc[0].max(0)[0].tolist()
    return avg_enc, max_enc,


#ICAAD_data = pd.read_csv(DATA_PATH + filename, index_col=0)
#ICAAD_data = ICAAD_data[ICAAD_data['category'].isin(['Evidence.no.SADV', 'Evidence.of.SA'])]
#train_df = ICAAD_data.sample(20, random_state=42)
#valid_df = ICAAD_data.drop(train_df.index).sample(frac=0.5, random_state=42)
#test_df = ICAAD_data.drop(train_df.index.append(valid_df.index))

data_lm = TextLMDataBunch.from_df('../data/', train_df=ICAAD_data[:2000], valid_df=ICAAD_data[2500:2600], bs=32)
#data_lm = TextLMDataBunch.from_csv(DATA_PATH, filename, text_cols='text')
#data_clas = TextClasDataBunch.from_csv(DATA_PATH, filename, text_cols='text', label_cols='category', vocab=data_lm.train_ds.vocab, bs=32, valid_pct=0.95)

#data_clas = TextClasDataBunch.from_df(path=DATA_PATH, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols='text', label_cols='category', vocab=data_lm.train_ds.vocab, bs=32)

#learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1, drop_mult=0.5)
learn = language_model_learner(data_lm, pretrained_fnames=[WIKI_MODEL, WIKI_VOCAB], drop_mult=0.5)

#lrf = learn.lr_find()

if train:
    print('*** Finetune Language Model ***')
    learn.freeze()
    learn.fit_one_cycle(1, 1e-3, div_factor=20, wd=1e-7)
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3, div_factor=20, wd=1e-7)

    learn.save_encoder('ICAAD')
else:
    learn.load_encoder('ICAAD')

pdb.set_trace()
#learn.save_encoder('ICAAD_enc')


sent1 = 'sexual assault'
enc1 = encode(learn, sent1)

sent2 = 'the girl was raped'
enc2 = encode(learn, sent2)

sent3 = 'apples are fruits'
enc3 = encode(learn, sent3)

sent4 = 'apples'
enc4 = encode(learn, sent4)
sent5 = 'rape'
enc5 = encode(learn, sent5)
sent6 = 'incest'
enc6 = encode(learn, sent6)
sent7 = 'murder'
enc7 = encode(learn, sent7)

word = 'girl'
enc2 = encode(learn, word)

cosine(enc1, enc2)
cosine(enc2, enc3)
cosine(enc1, enc3)
cosine(enc4, enc5)
cosine(enc6, enc5)
cosine(enc6, enc4)
cosine(enc6, enc4)
cosine(enc6, enc7)
cosine(enc5, enc7)
cosine(enc4, enc7)

learn.model.reset()
word_vocab, _ = learn.data.one_item(word)
sent3_enc = learn.model(torch.tensor(sent3_vocab))[-1][-1]
sent3_emb = sent3_enc[0].mean(0).tolist()


pdb.set_trace()
#from lang_model_utils import load_lm_vocab
#vocab = data_lm.vocab

itos = pickle.load(Path('../data/models/{}.pkl'.format(WIKI_VOCAB)).open('rb'))
stoi = defaultdict(lambda:-1, {v:k for k,v in enumerate(itos)})

wgts = torch.load('../data/models/{}.pth'.format(WIKI_MODEL), map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])

lang_model = learn.model

ary = [1860]
input_ary = torch.tensor(np.expand_dims(np.array(ary), -1)).cuda()
lang_model.reset()
hidden_states = lang_model(input_ary)[-1][-1]
hidden_states.mean(0)




text_classifier = text_classifier_learner(data_clas, drop_mult=0.5)
text_classifier.load_encoder('ICAAD_enc')

print('*** Train Classifier ***')
text_classifier.fit_one_cycle(3, 1e-2)
text_classifier.freeze_to(-2)
text_classifier.fit_one_cycle(5, slice(5e-3/2., 5e-3))
text_classifier.unfreeze()
text_classifier.fit_one_cycle(5, slice(2e-3/100, 2e-3))

#values, indices = predictions.max(1)

for ind, df in enumerate([valid_df]):
    if ind == 1:
        predictions = text_classifier.get_preds('test', ordered=True)[0]
    else:
        predictions = text_classifier.get_preds(ordered=True)[0]

    #pdb.set_trace()
    values, indices = predictions.max(1)

    #mapping = {'Evidence.no.SADV':0, 'Evidence.of.DV':1, 'Evidence.of.SA':2}
    mapping = {'Evidence.no.SADV':0, 'Evidence.of.SA':1}
    targetLabel = df.replace({'category': mapping}).category

    evaluation = Evaluation(targetLabel.tolist(), prediction=indices.tolist())
    evaluation.computeMeasures()
    evaluation.confusionMatrix()

    print('Accuracy: ' + str(evaluation.accuracy))
    print('Recall: ' + str(evaluation.recall))
    print('Precision: ' + str(evaluation.precision))
    print(evaluation.confusionMatrix)

    evaluation.createTags()
    df['tags'] = evaluation.tags

    pd.set_option('display.max_colwidth', 500)
    viewer = Viewer('ulmfit')
    viewer.use_classificationResults('ulmfit', ['no evidences'], df.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'docID', 'label']), 'ulmfit', 'wiki103', 0.5, evaluation, 0.11)


pdb.set_trace()
