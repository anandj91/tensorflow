import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os

'''
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/tmp/tf_graphs/xla_dump"
'''

class Model:
    def __init__(self, lr, num_class, dtype):
        self.num_class = num_class
        self.optimizer = tf.optimizers.Adam(lr)
        self.dtype = dtype

        self.W = self._get_weight([784, self.num_class], "weight")
        self.b = self._get_weight([self.num_class], "bias")
        self.weights = [self.W, self.b]

    def _get_weight(self, shape, name):
        return tf.Variable(tf.cast(np.ones(shape), dtype=self.dtype), name=name, trainable=True, dtype=self.dtype)

    @tf.function(experimental_compile=True)
    def forward(self, x):
        x = tf.cast(x, dtype=self.dtype)
        out = (tf.matmul(x, self.W) + self.b)
        out = tf.cast(out, dtype=tf.float32)
        return tf.nn.softmax(out)

    def loss(self, y_true, y_pred):
        return tf.losses.categorical_crossentropy(y_true, y_pred)

    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.weights))

batch_size = 32  #@param {type: "number"}
learning_rate = 0.001  #@param {type: "number"}
num_epochs = 5 #@param {type: "number"}
dataset = "mnist"
num_class = 10

model = Model(learning_rate, num_class, tf.cus);

def train_step(model, x, y):
    with tf.GradientTape() as tape:
        pred = model.forward(x)
        loss = model.loss(y, pred)
    grads = tape.gradient(loss, model.weights)
    model.update(grads)
    return grads

x = tf.cast(np.ones((batch_size, 784)), dtype=tf.float32)
y = tf.cast(np.ones((batch_size, 10)), dtype=tf.float32)
print(train_step(model, x, y))
