import tensorflow as tf
import osHelper
import math
from sklearn.metrics import precision_score, recall_score

class NeuralNet:

    def __init__(self, input_size=None, output_size=None):
        self.input_size = input_size
        self.X = tf.placeholder(tf.int32, [None, self.input_size], name='INPUT_X')
        self.output_size = output_size
        self.Y_ = tf.placeholder(tf.int64, [None, self.output_size], name='INPUT_Y')
        self.pkeep = tf.placeholder(tf.float32, shape=(), name='pkeep')


    def buildNeuralNet(self, vocab_size=None, filter_sizes=[3,4,5], embedding_size=300, num_filters=128, learning_rate=1e-3):

        self.l2_loss=tf.constant(0.0, name='l2_loss')

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='embedding_weights')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.X, name='embedded_chars')
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1, name='embedded_chars_expanded')

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter")
                b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_filter")
                self.conv = tf.nn.conv2d(self.embedded_chars_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(self.conv, b_filter), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                tf.summary.histogram("weights", W_filter)
                tf.summary.histogram("biases", b_filter)
                tf.summary.histogram("relu activation", h)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3, name='pooled_outputs')
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name='feature_vector')

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.pkeep)

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

            self.b = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(self.b)
            self.Ylogits = tf.nn.xw_plus_b(self.h_drop, W, self.b, name="scores")
            #self.Y = tf.argmax(self.Ylogits, 1, name='Y')
            self.predictions = tf.argmax(self.Ylogits, 1, name='predictions')
            self.probability = tf.reduce_max(tf.nn.softmax(self.Ylogits), 1, name='probability')

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Ylogits, labels=tf.stop_gradient(self.Y_), name="cross_entropy_with_logits")
            l2_reg_lambda = 3.0
            self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * self.l2_loss
            loss_summary = tf.summary.scalar("loss", self.loss)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
        lr_summary = tf.summary.scalar("learning_rate", learning_rate)
        self.train_step = optimizer.minimize(self.loss)
        tf.add_to_collection("train_step", self.train_step)
        self.getAccuracy()
        #self.getConfusionMatrix()
        #self.gradients(optimizer)

    def setupSummaries(self, graph, directory):
        #self.summaryWriter = tf.summary.FileWriter(directory)
        #self.summaryWriter.add_graph(graph)
        #tf.add_to_collection("summaryWriter", self.summaryWriter)

        #loss_summary = tf.summary.scalar("loss", self.loss)

        ##self.evaluationSummary()
        ##self.gradientSummary()
        ##self.imageSummary = tf.summary.Image(
        self.summaries = tf.summary.merge_all()
        tf.add_to_collection("summaries", self.summaries)

    def setSaver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def setSummaryWriter(self, path, graph):
        states = ['train', 'test']
        for state in states:
            state_path = path + state
            osHelper.deleteFolderContent(state_path)
            summary_writer = tf.summary.FileWriter(state_path, graph)
            setattr(self, '_'.join([state, 'summary']), summary_writer)

    def evaluationSummary(self):
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        #lr_summary = tf.summary.scalar("learning rate", self.learning_rate)
        #self.summary = tf.summary.merge([loss_summary, acc_summary, lr_summary])
        self.summary = tf.summary.merge([loss_summary, acc_summary])

    def saveSummary(self, summary, step):
        self.summaryWriter.add_summary(summary, step)

    def writeSummary(self, summary, step, phase='train'):
        if phase=='train':
            self.train_summary.add_summary(summary, step)
        elif phase=='test':
            self.test_summary.add_summary(summary, step)
        else:
            print 'WARNING: Phase %s in write summary is not known'


    def gradients(self, optimizer):
        self.grads_and_vars = optimizer.compute_gradients(self.loss)


    def getAccuracy(self):
        is_correct = tf.equal(self.predictions, tf.argmax(self.Y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")

    def getConfusionMatrix(self):
        self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.Y_,1), self.Y)


    def learningRate(self, step, max_lr=0.003, min_lr=0.0001, decay_speed=2000):
        return min_lr + (max_lr - min_lr) * math.exp(-step/decay_speed)

    def gradientSummary(self):
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries = tf.summary.merge(grad_summaries)

    def imageSummary(self, inputImages):
        for image in inputImages:
            tf.summary.image('image', image)


    def variable_summaries(self, var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def saveCheckpoint(self, session, path, step):
        self.saver.save(session, path, global_step=step)
        #self.summaryWriter.close()

    def loadCheckpoint(self, graph, session, checkpoint_path):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        self.saver.restore(session, checkpoint_file)

        self.X = graph.get_operation_by_name("INPUT_X").outputs[0]
        self.Y_ = graph.get_operation_by_name("INPUT_Y").outputs[0]
        #self.Y = graph.get_operation_by_name("Y").outputs[0]
        self.pkeep = graph.get_operation_by_name("pkeep").outputs[0]
        #self.learning_rate = graph.get_operation_by_name("learning_rate").outputs[0]

        #self.summaryWriter = tf.summary.FileWriter(checkpoint_path)
        self.global_step = graph.get_operation_by_name('global_step').outputs[0]
        self.summaries = graph.get_collection("summaries")[0]

        self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        self.Ylogits = graph.get_operation_by_name("output/scores").outputs[0]
        self.probability = graph.get_operation_by_name("output/probability").outputs[0]

        self.h_pool = graph.get_operation_by_name("pooled_outputs").outputs[0]
        self.h_pool_flat = graph.get_operation_by_name("feature_vector").outputs[0]
        self.train_step = tf.get_collection("train_step")[0]



