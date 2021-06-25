import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
#os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/home/chenhao/projects/xla_dump/mlp"

class Model:
    def __init__(self, lr, num_class, dtype):
        self.num_class = num_class
        self.optimizer = tf.optimizers.Adam(lr)
        self.dtype = dtype

        self.W = self._get_weight([5, 5], "weight")
        self.b = self._get_weight([5], "bias")
        self.weights = [self.W, self.b]

    def _get_weight(self, shape, name):
        return tf.Variable(tf.cast(np.ones(shape), dtype=self.dtype), name=name, trainable=True, dtype=self.dtype)

    @tf.function(experimental_compile=True)
    def forward(self, x):
        x = tf.cast(x, dtype=self.dtype)
        out = (tf.matmul(x, self.W) + self.b)
        out = tf.cast(out, dtype=tf.float32)
        return out

    def loss(self, y_true, y_pred):
        return tf.losses.categorical_crossentropy(y_true, y_pred)

    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.weights))

batch_size = 32  #@param {type: "number"}
learning_rate = 0.001  #@param {type: "number"}
num_epochs = 5 #@param {type: "number"}
dataset = "mnist"
num_class = 10

model = Model(learning_rate, num_class, tf.float64)
#model = Model(learning_rate, num_class, tf.cus)

def train_step(model, x, y):
    with tf.GradientTape() as tape:
        pred = model.forward(x)
    grads = tape.gradient(pred, model.weights)
    return grads

with open('example/data.npy', 'rb') as f:
    x = np.load(f)
    y = np.load(f)

grad = train_step(model, x, y)
print("W:", model.W)
print("B:", model.b)
print("dW:", grad[0])
print("db:", grad[1]) 
