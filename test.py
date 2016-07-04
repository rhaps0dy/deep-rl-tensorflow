import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

with tf.Session() as sess, tf.variable_scope('shit'):
  optimizer = tf.train.RMSPropOptimizer(0.025, momentum=0.95, epsilon=0.01, use_locking=True)
  vec_global = tf.get_variable('wg', [2,2], tf.float32, initializer=initializers.xavier_initializer(), trainable=False)
  vec_local = tf.get_variable('wl', [2,2], tf.float32, trainable=False)
  loss = tf.reduce_sum(tf.square(1-vec_local))
  grad_local = tf.gradients(loss, [vec_local])
  grad_global = [(grad_local[0], vec_global)]
  opt_op = optimizer.apply_gradients(grad_global)
  copy_op = vec_local.assign(vec_global)

  tf.initialize_all_variables().run()
  print sess.run([vec_global, vec_local])

  while True:
    print sess.run([copy_op])
    print ""
    print sess.run([vec_global, vec_local])
    print ""
    print sess.run([opt_op, vec_global, vec_local])
    print ""
    print sess.run([vec_global, vec_local])
    print "============================================"*4
    raw_input()
