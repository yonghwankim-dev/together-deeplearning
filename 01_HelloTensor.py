import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")

#start a TF session
sess = tf.Session()

#run the op and get result
print(sess.run(hello))
