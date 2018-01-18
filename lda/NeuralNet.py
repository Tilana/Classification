import tensorflow as tf
import osHelper
import math
from sklearn.metrics import precision_score, recall_score

class NeuralNet:


    def __init__(self, input_size=None, output_size=None):
        self.input_size = input_size
        self.X = tf.placeholder(tf.int32, [None, self.input_size], name='X')
        self.output_size = output_size
        self.Y_ = tf.placeholder(tf.int64, [None, self.output_size], name='Y_')
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.pkeep = tf.placeholder(tf.float32, shape=(), name='pkeep')
        self.step = tf.placeholder(tf.float32, shape=(), name='step')


    def buildNeuralNet(self, nnType='cnn', vocab_size=None, hidden_layer_size=100, optimizerType='GD', sequence_length=None, filter_sizes=[3,4,5], secondLayer=False):
        self.nnType = nnType
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        if nnType=='multi':
            self.multiLayerNN(hidden_layer_size)
        elif nnType =='cnn':
            self.cnn(filter_sizes=filter_sizes, secondLayer=secondLayer)
        else:
            self.oneLayerNN()
        self.crossEntropy()
        self.optimizer(optimizerType)
        self.getAccuracy()
        self.getConfusionMatrix()
        self.trainStep()
        self.evaluationSummary()
        self.gradients()
        self.gradientSummary()


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
        loss_summary = tf.summary.scalar("loss", self.cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        lr_summary = tf.summary.scalar("learning rate", self.learning_rate)
        self.summary = tf.summary.merge([loss_summary, acc_summary, lr_summary])

    def writeSummary(self, summary, step, phase='train'):
        if phase=='train':
            self.train_summary.add_summary(summary, step)
        elif phase=='test':
            self.test_summary.add_summary(summary, step)
        else:
            print 'WARNING: Phase %s in write summary is not known'

    def oneLayerNN(self):
        self.W = tf.Variable(tf.zeros([self.input_size, self.output_size]))
        self.b = tf.Variable(tf.zeros([self.output_size]))
        self.Ylogits = tf.matmul(self.X, self.W) + self.b
        self.Y = tf.nn.softmax(self.Ylogits)

    def multiLayerNN(self, hidden_layer_size=100):
        self.W1 = tf.Variable(tf.truncated_normal([self.input_size, hidden_layer_size]))
        self.W2 = tf.Variable(tf.truncated_normal([hidden_layer_size, self.output_size]))
        self.b1 = tf.Variable(tf.ones([hidden_layer_size])/10)
        self.b2 = tf.Variable(tf.zeros([self.output_size]))

        self.Y1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        self.Y1d = tf.nn.dropout(self.Y1, self.pkeep)

        self.Ylogits = tf.matmul(self.Y1d, self.W2) + self.b2
        self.Y = tf.nn.softmax(self.Ylogits)

    def cnn(self, embedding_size=300, filter_sizes=[4,5,6,7], num_filters=128, secondLayer=False):
        self.l2_loss=tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0, seed=42), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.X)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter")
            b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_filter")
            conv = tf.nn.conv2d(self.embedded_chars_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

        if secondLayer:
            conv2 = tf.layers.conv2d(inputs=self.h_pool, filters=num_filters, kernel_size=[2,2], padding="same", activation=tf.nn.relu)
            self.h_pool_flat = tf.reshape(conv2, [-1, num_filters_total])
        else:
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.pkeep)

        W = tf.get_variable("W", shape=[num_filters_total, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

        self.b = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="b")
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(self.b)
        self.Ylogits = tf.nn.xw_plus_b(self.h_drop, W, self.b, name="scores")
        self.Y = tf.argmax(self.Ylogits, 1, name='Y')
        self.predictions = tf.argmax(self.Ylogits, 1, name='predictions')
        self.probability = tf.reduce_max(tf.nn.softmax(self.Ylogits), 1, name='probability')


    def crossEntropy(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y_)
        if self.nnType=='cnn':
            l2_reg_lambda = 0.0
            self.cross_entropy = tf.reduce_mean(cross_entropy) #+ l2_reg_lambda * self.l2_loss
        else:
            self.cross_entropy = tf.reduce_mean(cross_entropy)*100

    def optimizer(self, optimizerType='GD', learning_rate=1e-3, decay=False):
        if decay:
            learning_rate = self.learning_rate
        if optimizerType == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizerType == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            print 'Warning: unkown Optimizer'

    def gradients(self):
        self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy)

    def trainStep(self, summary=1):
        self.train_step = self.optimizer.minimize(self.cross_entropy) #, name="train_step")

    def getAccuracy(self):
        if self.nnType=='cnn':
            is_correct = tf.equal(self.Y, tf.argmax(self.Y_,1))
        else:
            is_correct = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


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

    def loadCheckpoint(self, graph, session, checkpoint_path):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        self.saver.restore(session, checkpoint_file)

        self.X = graph.get_operation_by_name("X").outputs[0]
        self.Y_ = graph.get_operation_by_name("Y_").outputs[0]
        self.Y = graph.get_operation_by_name("Y").outputs[0]
        self.pkeep = graph.get_operation_by_name("pkeep").outputs[0]
        self.learning_rate = graph.get_operation_by_name("learning_rate").outputs[0]
        self.step = graph.get_operation_by_name("step").outputs[0]

        self.predictions = graph.get_operation_by_name("predictions").outputs[0]
        self.Ylogits = graph.get_operation_by_name("scores").outputs[0]
        self.probability = graph.get_operation_by_name("probability").outputs[0]

        self.nnType = 'cnn'

        self.crossEntropy()
        self.optimizer(optimizerType='GD')
        #self.getAccuracy()
        #self.getConfusionMatrix()
        self.trainStep()
