from __future__ import division
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pdb

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Visualization
# ==================================================

def plotDocumentLengthDistribution(documents):
    documentLengths = [len(text.split(" ")) for text in documents]
    logbins = np.max(documentLengths)*(np.logspace(0,1, num=50) - 1)/9
    hist, bins = np.histogram(documentLengths, bins=logbins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, log=1)
    plt.xlabel('Number of words in document')
    plt.ylabel('log Frequency')
    plt.title('ICAAD - Document length')
    plt.show()


def boxplot(documents):
    documentLengths = [len(text.split(" ")) for text in documents]
    plt.boxplot(documentLengths)
    plt.yscale('log')
    plt.show()

# Data Exploration
# ==================================================

def summarize(y, name, categoryOfInterest):
    num = len(y)
    numPos = len([elem for elem in y if elem[1]==categoryOfInterest])
    dataLength= 'Number of data in %s set:  %i' % (name, num)
    distribution = 'Positive samples: %i --> %i Percent' % (numPos, numPos/(num/100.0))
    #return '\n'.join([dataLength, distribution])
    return [dataLength, distribution]



# Data Preparation
# ==================================================
import pandas as pd

dataset = 'ICAAD'
id_name = 'SA'
path = '../data/ICAAD/sentences_ICAAD.csv'
target = 'category'
categoryOfInterest = 'Evidence.of.{:s}'.format(id_name)
negCategory = 'Evidence.no.SADV'

dataset = 'Manifesto'
path = '../data/Manifesto/manifesto_United Kingdom.csv'
target = 'cmp_code'
id_name = 'cmp_code'
categoryOfInterest = 110

dataset = 'UPR'
path = '../data/UPR/UPR_DATABASE.csv'
target = 'label'
id_name = 'UPR'
categoryOfInterest = 'Minorities'

dataset = 'ICAAD'
id_name = 'DV'
path = '../data/ICAAD/ICAAD_evidenceSummary.pkl'
textCol = 'evidenceText_' + id_name
target = 'Domestic.Violence.Manual'



# Load data
print("Loading data...")
#data = pd.read_csv(path)

data = pd.read_pickle(path)
data = data.dropna(subset=[textCol])

#pdb.set_trace()
#data = data.rename(columns = {'Unnamed: 0': 'id'})
#data = data.rename(columns = {'Recommendation': 'text'})

#posSample = data[data[target]==categoryOfInterest]
#negSample = data[data[target] == negCategory].sample(len(posSample))
#data = pd.concat([posSample, negSample])

# Randomly split data in training and test set
labels = data[target].tolist()
y = pd.get_dummies(labels).values

#x_train, x_dev, y_train, y_dev = train_test_split(data.sentence, y, test_size=0.3, random_state=200)
x_train, x_dev, y_train, y_dev = train_test_split(data[textCol], y, test_size=0.3, random_state=200)
trainDocs = data.loc[x_train.index].id.unique()
testDocs = data.loc[x_dev.index].id.unique()
indices = pd.DataFrame([trainDocs, testDocs], index=['train', 'test'])

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_train])
infoSentenceLength = 'Maximal sentence Length: {:d}'.format(max_document_length)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_dev = np.array(list(vocab_processor.transform(x_dev)))

vocabulary = vocab_processor.vocabulary_._mapping

infoDatabase = "Database: {:s}".format(dataset)
infoCOI = "Category of Interest: {:s}".format(str(categoryOfInterest))
infoClassNumber =  "Number of categories: {:d}".format(len(data[target].unique()))
infoTarget = "Target: {:s}".format(target)
infoVocab = "Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))
infoSplit = "Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev))

infoData = [infoDatabase, infoCOI]
infoSentences = [infoSentenceLength, infoVocab]
infoTarget = [infoTarget, infoClassNumber]

infoTrain = summarize(y_train, 'training', categoryOfInterest)
infoTest = summarize(y_dev, 'dev', categoryOfInterest)

pdb.set_trace()



# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1], vocab_size=len(vocab_processor.vocabulary_), embedding_size=FLAGS.embedding_dim, filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), num_filters=FLAGS.num_filters, l2_reg_lambda=FLAGS.l2_reg_lambda)


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        # Output directory for models and summaries
        result_folder = '_'.join([dataset, id_name])
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", result_folder))
        print("Writing to {}\n".format(out_dir))

        # Training Testing indices
        indices.to_csv(os.path.join(out_dir, 'trainTest_split.csv'))

        # Data Summary
        dataInfo = tf.convert_to_tensor([infoData, infoTarget, infoSentences, infoTrain, infoTest])
        text_summary = tf.summary.text(dataset, dataInfo)
        sentenceInfo = tf.convert_to_tensor(infoSentenceLength)
        text_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "info"), sess.graph)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        prec_summary = tf.summary.scalar("precision", cnn.precision)
        rec_summary = tf.summary.scalar("recall", cnn.recall)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, prec_summary, rec_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, prec_summary, rec_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        info_summary = sess.run(text_summary)
        text_summary_writer.add_summary(info_summary)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            print 'Train Step'
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy, precision, recall = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall], feed_dict)
            time_str = datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, rec {:g}".format(time_str, step, loss, accuracy, precision, recall))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            print 'Dev Step'
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, precision, recall = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall], feed_dict)

            time_str = datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, rec: {:g}".format(time_str, step, loss, accuracy, precision, recall))
            if writer:
                writer.add_summary(summaries, step)



        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            #pdb.set_trace()
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

