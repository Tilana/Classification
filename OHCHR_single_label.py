import sys
sys.path.insert(0, '/home/natalie/Documents/Huridocs-Code/Classification/lda')
from fastai.text import *
from fastai.vision import *
from lda.Evaluation import Evaluation
from lda.Viewer import Viewer
import pandas as pd
import pdb

DATA_PATH = '../data/'
filename = 'OHCHR/uhri_universalIssues.csv'

CATEGORY = 'B'

train_lm = 0
train_cls = 1

WIKI_MODEL = 'wt103/fwd_wt103'
WIKI_VOCAB = 'wt103/itos_wt103'

data = pd.read_csv(DATA_PATH + filename)

def extractCategory(themes, category='B'):
    themes = themes.split('\n- ')
    themes[0] = themes[0][2:]
    themes[-1] = themes[-1][:-1]
    category_themes = [theme for theme in themes if theme[0]==category]
    if len(category_themes)>1:
        if 'effective remedy' in category_themes[0]:
            return category_themes[1]
    return category_themes[0]

data[CATEGORY] = data['Themes'].apply(extractCategory, CATEGORY)
data[CATEGORY].value_counts()

data.rename(columns={CATEGORY: 'label', 'Annotation': 'text'}, inplace=True)
data = data[['text', 'label']]

if train_lm:
    lm_train_df = data.sample(22000, random_state=42)
    lm_valid_df = data.drop(lm_train_df.index)

    data_lm = TextLMDataBunch.from_df(DATA_PATH, train_df=lm_train_df, valid_df=lm_valid_df, bs=32, text_cols='text', label_cols='label')

    learn = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=[WIKI_MODEL, WIKI_VOCAB], drop_mult=0.3, wd=0.1)

    print('*** Finetune Language Model ***')
    learn.freeze()
    learn.fit_one_cycle(1, 1e-3)
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(1e-5, 1e-3))

    learn.save_encoder('UHRI')
else:
    data_lm = TextLMDataBunch.load(DATA_PATH)

MIN_NR = 100
MAX_NR = len(data)

themes = data['label'].value_counts()
valid_themes = themes[(themes >= MIN_NR) & (themes <= MAX_NR)].keys().tolist()
data = data[data['label'].isin(valid_themes)]

data = data.groupby('label', as_index=False).apply(lambda x: x.sample(MIN_NR, random_state=42))

NR_TRAIN = 900
train_df = data.sample(NR_TRAIN, random_state=42)
valid_df = data.drop(train_df.index).sample(frac=0.5, random_state=42)
test_df = data.drop(train_df.index.append(valid_df.index))

pdb.set_trace()

data_clas = TextClasDataBunch.from_df(DATA_PATH, train_df=train_df, valid_df=valid_df, test_df=test_df, label_cols='label', text_cols='text', bs=4, vocab=data_lm.vocab)
#text_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.1, wd=0.1, metrics=[fbeta])
text_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.1, wd=0.1)
text_classifier.load_encoder('UHRI')

if train_cls:
    print('*** Train Classifier ***')
    text_classifier.freeze()
    text_classifier.fit_one_cycle(1, 1e-2)
    text_classifier.freeze_to(-2)
    text_classifier.fit_one_cycle(1, slice(5e-3/2., 5e-3))
    text_classifier.freeze_to(-3)
    text_classifier.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))

    pdb.set_trace()
    text_classifier.unfreeze()
    text_classifier.fit_one_cycle(3, slice(1e-3/(2.6**4), 1e-3))
    #text_classifier.fit_one_cycle(15, slice(1e-3/(2.6**4), 1e-3))

    text_classifier.save('UHRI_AWD_LSTM_CLASSIFIER_{}'.format(CATEGORY))

else:
    text_classifier.load('UHRI_AWD_LSTM_CLASSIFIER_{}'.format(CATEGORY))

predictions = test_df.text.apply(text_classifier.predict)

category, _, probability = zip(*predictions.values)
pred_category = [x[0].obj for x in predictions.values]
probabilities = [max(x[2].tolist()) for x in predictions.values]

target_label = test_df['label'].tolist()
evaluation = Evaluation(target_label, pred_category, 'macro')
evaluation.computeMeasures()
evaluation.confusionMatrix(labels=data_clas.classes)

print('Accuracy: ' + str(evaluation.accuracy))
print('Recall: ' + str(evaluation.recall))
print('Precision: ' + str(evaluation.precision))
print(evaluation.confusionMatrix)

evaluation.createTags()
test_df['pred_label'] = pred_category
test_df['tags'] = evaluation.tags
test_df['probability'] = probabilities

pd.set_option('display.max_colwidth', 500)
viewer = Viewer('uhri')
test_df.sort_values(['tags', 'probability'], inplace=True, ascending=[True, False])
info = {'name': 'uhri', 'Number Train Data': len(train_df), 'Number Validation Data': len(valid_df), 'Number of Test Data': len(test_df), 'Classes': data_clas.classes}
viewer.classification(info, test_df, evaluation)

pdb.set_trace()
