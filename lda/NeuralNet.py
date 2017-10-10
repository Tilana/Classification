import tensorflow as tf
import osHelper
import math

class NeuralNet:

    def __init__(self, input_size, output_size, multilayer=1, hidden_layer_size=100, optimizerType='GD'):
        self.input_size = input_size
        self.output_size = output_size
        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y_ = tf.placeholder(tf.float32, [None, self.output_size])
        self.step = tf.placeholder(tf.float32, shape=(), name='init')
        self.learning_rate = tf.placeholder(tf.float32, shape=())

    def buildNeuralNet(self, multilayer, hidden_layer_size, optimizerType):
        if multilayer:
            self.multiLayerNN(hidden_layer_size)
        else:
            self.oneLayerNN()
        self.crossEntropy()
        self.optimizer(optimizerType)
        self.getAccuracy()
        self.trainStep()
        self.evaluationSummary()
        self.gradients()
        self.gradientSummary()

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
        self.Ylogits = tf.matmul(self.Y1, self.W2) + self.b2
        self.Y = tf.nn.softmax(self.Ylogits)

    def crossEntropy(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(cross_entropy)*100

    def optimizer(self, optimizerType='GD', learning_rate=0.003, decay=True):
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
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def getAccuracy(self):
        is_correct = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def learningRate(self, step, max_lr=0.003, min_lr=0.0001, decay_speed=1500):
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
