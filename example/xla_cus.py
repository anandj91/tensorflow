import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/tmp/tf_graph"
import signal

import numpy as np
import tensorflow as tf


npa = np.arange(1, 10, 1.3) # 1, 2.3, 3.6, 4.9, 6.2, 7.5, 8.8
a = tf.cast(npa, tf.float32)

@tf.function(experimental_compile=True)
def model_fn(a):
    b = tf.cast(a, tf.cus)
    c = tf.add(b, b)
    d = tf.cast(c, tf.float32)
    return d

print(model_fn(a))
