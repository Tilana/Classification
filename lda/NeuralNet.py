#import tensorflow as tf

class NeuralNet:

    def __init__(self):
        self.session = tf.InteractiveSession()

    def createVariables(self, numberFeatures, numberCategories):
        self.x = tf.placeholder(tf.float32, [None, numberFeatures])
        self.y_ = tf.placeholder(tf.float32, [None, numberCategories])
        self.W = tf.Variable(tf.zeros([numberFeatures, numberCategories]))
        self.b = tf.Variable(tf.zeros([numberCategories]))

    def initializeVariables(self):
        self.session.run(tf.initialize_all_variables())

    def createSoftmax(self):
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

    def createLossfunction(self):
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

    def createTrainStep(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

    def setup(self, numberFeatures, numberCategories):
        self.createVariables(numberFeatures, numberCategories)
        self.initializeVariables()
        self.createSoftmax()
        self.createLossfunction()
        self.createTrainStep()

    def train(self, trainData, trainTarget, numberTrainingSteps=1000):
        for step in range(numberTrainingSteps):
            self.session.run(self.train_step, feed_dict = {self.x: trainData, self.y_: trainTarget})

    def predict(self, testData):
        prediction = tf.argmax(self.y,1)
        return self.session.run(prediction, feed_dict = {self.x: testData})




