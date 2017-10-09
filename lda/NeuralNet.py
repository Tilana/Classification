import tensorflow as tf

class NeuralNet:

    def __init__(self, input_size, output_size, multilayer=1, hidden_layer_size=100, optimizerType='GD'):
        self.input_size = input_size
        self.output_size = output_size
        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y_ = tf.placeholder(tf.float32, [None, self.output_size])
        self.step = tf.placeholder(tf.float32, shape=(), name='init')

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

    def setSummaryWriter(self, path, graph):
        self.train_summary = tf.summary.FileWriter(path + 'train', graph)
        self.test_summary = tf.summary.FileWriter(path + 'test', graph)

    def evaluationSummary(self):
        loss_summary = tf.summary.scalar("loss", self.cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge([loss_summary, acc_summary])

    def writeSummary(self, summary, step, phase='train'):
        if phase=='train':
            self.train_summary.add_summary(summary, step)
        elif phase=='test':
            self.test_summary.add_summary(summary, step)
        else:
            print 'WARNING: Phase %s in write summary is not known'

    def oneLayerNN(self):
        self.W = tf.Variable(tf.zeros([self.input_size, self.output_size]))
        self.b = tf.Variable(tf.ones([self.output_size])/10)
        self.Ylogits = tf.matmul(self.X, self.W) + self.b
        self.Y = tf.nn.softmax(self.Ylogits)

    def setSession(self, sess):
        self.sess = sess

    def initializeVariables(self):
        self.sess.run(tf.global_variables_initializer())

    def multiLayerNN(self, hidden_layer_size=100):
        self.W1 = tf.Variable(tf.truncated_normal([self.input_size, hidden_layer_size]))
        self.W2 = tf.Variable(tf.truncated_normal([hidden_layer_size, self.output_size]))
        self.b1 = tf.Variable(tf.ones([hidden_layer_size])/10)
        self.b2 = tf.Variable(tf.ones([self.output_size])/10)

        self.Y1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        self.Ylogits = tf.matmul(self.Y1, self.W2) + self.b2
        self.Y = tf.nn.softmax(self.Ylogits)

    def crossEntropy(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(cross_entropy)*100

    def optimizer(self, optimizerType='GD', learning_rate=0.003):
        if optimizerType == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizerType == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            print 'Warning: unkown Optimizer'

    def trainStep(self, summary=1):
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def getAccuracy(self):
        is_correct = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



