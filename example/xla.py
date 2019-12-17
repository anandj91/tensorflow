import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'

import numpy as np
import tensorflow as tf

from tensorflow.contrib.compiler import xla

inp = np.random.rand(10, 10)

def model_fn(x):
    y = tf.sigmoid(tf.matmul(x, x))
    return y

x = tf.placeholder(tf.float32, shape=(10, 10))
[y] = xla.compile(model_fn, inputs=[x])

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: inp}))
