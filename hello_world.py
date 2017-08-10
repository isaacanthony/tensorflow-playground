import tensorflow as tf

hello = tf.constant('Hello, World!')

with tf.Session() as sess:
  print(sess.run(hello))
