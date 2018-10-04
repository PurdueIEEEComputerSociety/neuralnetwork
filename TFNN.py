import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#	Can use these to display the data if we want
# import matplotlib.pyplot as plt
# import numpy as np
# import random as ran

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#	Here one-hot is referring to the output encoding, not the input encoding

#	Arbitrary by 784 size (no fixed number of examples that'll be run through it)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])	#	Why float here? These are bools for one-hot, no?

#	Weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#	Classification function
y = tf.nn.softmax(tf.matmul(x,W) + b)	#	Softmax puts everything on P scale

#	Element-wise multiplication of true and predicted values
#	Take negative log to penalize confident incorrect predictions more heavily (log(0.01)=-2, -2*0=0, -2*1=-2)
loss_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#	Train and test on subset of data to conserve CPU
TRAIN_SIZE = 5500
TEST_SIZE = 1000

x_train, y_train = (mnist.train.images[:TRAIN_SIZE,:], mnist.train.labels[:TRAIN_SIZE,:])
x_test, y_test = (mnist.test.images[:TEST_SIZE,:], mnist.test.labels[:TEST_SIZE,:])
LEARNING_RATE = 0.01
TRAIN_STEPS = 2000

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	#	Init variables

	training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function)
	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	#	Perform the actual training
	for i in range(TRAIN_STEPS):
		sess.run(training, feed_dict={x: x_train, y_: y_train})
		if i % 100 == 0:
			print('Epoch {}, Accuracy={}, Loss={}'.format(i, sess.run(accuracy, feed_dict={x: x_test, y_: y_test}), sess.run(loss_function, feed_dict={x: x_train, y_: y_train})))


	sess.run()

