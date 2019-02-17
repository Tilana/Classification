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
train_cls = 0

WIKI_MODEL = 'wt103/fwd_wt103'
WIKI_VOCAB = 'wt103/itos_wt103'

data = pd.read_csv(DATA_PATH + filename)

def extractCategory(themes, category='B'):
    themes = themes.split('\n- ')
    themes[0] = themes[0][2:]
    themes[-1] = themes[-1][:-1]
    category_themes = [theme for theme in themes if theme[0]==category]
    return ';'.join(category_themes)

data[CATEGORY] = data['Themes'].apply(extractCategory, CATEGORY)
data[CATEGORY].value_counts()

data.rename(columns={CATEGORY: 'label', 'Annotation': 'text'}, inplace=True)
data = data[['text', 'label']]

if train_lm:
    lm_train_df = data.sample(22000, random_state=42)
    lm_valid_df = data.drop(lm_train_df.index)

    #data_lm = TextLMDataBunch.from_df(DATA_PATH, train_df=lm_train_df, valid_df=lm_valid_df, bs=32, text_cols='text', label_cols='label')
    #data_lm.save()
    data_lm = TextLMDataBunch.load(DATA_PATH)

    learn = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=[WIKI_MODEL, WIKI_VOCAB], drop_mult=0.3, wd=0.1)

    print('*** Finetune Language Model ***')
    learn.freeze()
    learn.fit_one_cycle(1, 1e-3, 1e-1) #, div_factor=20, wd=1e-7)
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(1e-5, 1e-3), wd=1e-1) # div_factor=20, wd=1e-7)
    #learn.fit_one_cycle(5, slice(1e-5,1e-3), div_factor=20, wd=1e-7)

    learn.save_encoder('UHRI')
else:
    data_lm = TextLMDataBunch.load(DATA_PATH)

MIN_NR = 100
MAX_NR = 150

categories = data['label'].str.split(';')
all_categories = sum(categories.tolist(), [])
category_counts = collections.Counter(all_categories)
valid_categories = [key for key, count in category_counts.items() if count >= MIN_NR]


is_in = [any(elem in valid_categories for elem in category) for category in categories]
data = data[is_in]

#pdb.set_trace()

#valid_categories = category_counts.most_common()
#valid_themes = themes[(themes >= MIN_NR) & (themes <= MAX_NR)].keys().tolist()

#themes = data['label'].value_counts()
#valid_themes = themes[(themes >= MIN_NR) & (themes <= MAX_NR)].keys().tolist()
#data = data[data['label'].isin(valid_themes)]
#data.shuffle(inplace=True, random_state=42)

def sample(data, no):
    if len(data) < no:
        return data.sample(len(data))
    return data.sample(no)

#pdb.set_trace()
single_label_data = data[data['label'].str.split(';').str.len() == 1]
balanced_data = single_label_data.groupby('label', as_index=False).apply(sample, MAX_NR) # .reset_index()
multi_label_data = data.drop(single_label_data.index)
data = pd.concat([balanced_data, multi_label_data])

NR_TRAIN = 4000
train_df = data.sample(NR_TRAIN, random_state=42)
valid_df = data.drop(train_df.index).sample(frac=0.8, random_state=42)
test_df = data.drop(train_df.index.append(valid_df.index))

data_clas = TextClasDataBunch.from_df(DATA_PATH, train_df=train_df, valid_df=valid_df, test_df=test_df, label_cols='label', text_cols='text', bs=4, vocab=data_lm.vocab, label_delim=';')


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

text_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.1, wd=0.1, metrics=[acc_02, f_score])

#text_classifier = RNN_Learner(model_data, text_model, opt_fn=opt_fn)
#text_classifier.crit = nn.CrossEntropyLoss()
#text_classifier.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
#text_classifier.clip=0.25
#text_classifier.metrics = [accuracy]
text_classifier.load_encoder('UHRI')

if train_cls:
    print('*** Train Classifier ***')
    text_classifier.freeze()
    text_classifier.fit_one_cycle(1, 1e-2)
    text_classifier.freeze_to(-2)
    text_classifier.fit_one_cycle(1, slice(5e-3/2., 5e-3))
    text_classifier.freeze_to(-3)
    text_classifier.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))

    text_classifier.save('UHRI_{}_multi'.format(CATEGORY))
    text_classifier.unfreeze()
    text_classifier.fit_one_cycle(3, slice(1e-3/(2.6**4), 1e-3))
    #text_classifier.fit_one_cycle(15, slice(1e-3/(2.6**4), 1e-3))

    text_classifier.save('UHRI_{}_multi'.format(CATEGORY))

else:
    text_classifier.load('UHRI_{}_multi'.format(CATEGORY))

#dir(text_classifier.data)

THRESHOLD = 0.3

preds, y = text_classifier.get_preds(DatasetType.Test, ordered=True)
#text_classifier.predict(test_df.iloc[500].text) #, metrics=[acc_02])

def get_pred_labels(predictions, classes, THRESHOLD=0.3):
    predictions = []
    for prediction in predictions:
        curr = []
        for ind, prob in enumerate(prediction.tolist()):
            if prob >= THRESHOLD:
                curr.append((classes[ind], prob))
        if len(curr) == 0:
            ind = np.argmax(prediction.tolist())
            curr.append((classes[ind], prediction.tolist()[ind]))
        predictions.append(curr)
    return predictions


#pdb.set_trace()

predictions = [[(data_clas.classes[ind], prob) for ind, prob in enumerate(prediction.tolist()) if prob >= THRESHOLD] for prediction in preds]


#pdb.set_trace()
#probability, pred_category2 = preds.max(1)
#predictions = test_df.text.apply(text_classifier.predict)

#predictions = test_df_copy.apply(text_classifier.predict, axis=1)
#predictions2 = test_df_copy.text.apply(text_classifier.predict)
#category, _, probability = zip(*predictions.values)
#pred_category = [x[0].obj for x in predictions.values]
#probabilities = [max(x[2].tolist()) for x in predictions.values]

#pred_category2 = [x[0].obj for x in predictions2.values]
#probabilities2 = [max(x[2].tolist()) for x in predictions2.values]


test_df['label'] = test_df['label'].str.split(';')
test_df['label'] = test_df.label.apply(sorted)
#test_df.label.sort_values()
#
target_label = test_df['label'].tolist()
pred_label = [[elem[0] for elem in prediction] for prediction in predictions]

test_df['pred_labels'] = predictions
evaluation = Evaluation(target_label, pred_label, classes=data_clas.classes)

#correct, missed, false = evaluation.setEvaluation(target_label, pred_label)

tags = evaluation.evalMultiLabel()
test_df['TP'] = tags['TP'].values
test_df['FN'] = tags['FN'].values
test_df['FP'] = tags['FP'].values

#evaluat = pd.DataFrame(evalua, index=['correct', 'missed', 'false']).transpose()


evaluation.computeMeasures()
#evaluation.confusionMatrix()
#evaluation.confusionMatrix.columns = data_clas.classes

evaluation.confusionMatrix = pd.DataFrame()

#print('Accuracy: ' + str(evaluation.accuracy))
#print('Recall: ' + str(evaluation.recall))
#print('Precision: ' + str(evaluation.precision))
#print(evaluation.confusionMatrix)

#evaluation.createTags()
#tests = pd.concat([test_df, evaluat], axis=1)
#test_df['correct'] = evaluation.multiLabelEval['correct']
#test_df['correct'] = evalua[0]

#test_df['pred_label'] = pred_category
#test_df['tags'] = evaluation.tags
#test_df['probability'] = probabilities


format_cols = ['pred_labels', 'label', 'FP', 'FN', 'TP']
for col in format_cols:
    test_df[col] = test_df[col].apply(lambda x: '\\n'.join(map(str, x)))

pdb.set_trace()
#test_df['pred_labels'] = test_df['pred_labels'].apply(lambda x: '\\n'.join(map(str, x)))

pd.set_option('display.max_colwidth', 500)
viewer = Viewer('uhri_multi')
#test_df.sort_values(['tags', 'probability'], inplace=True, ascending=[True, False])
info = {'name': 'uhri_multi', 'Number Train Data': len(train_df), 'Number Validation Data': len(valid_df), 'Number of Test Data': len(test_df), 'Threshold': THRESHOLD, 'Categories': dict(category_counts.most_common())}
viewer.classification(info, test_df, evaluation)

pdb.set_trace()
