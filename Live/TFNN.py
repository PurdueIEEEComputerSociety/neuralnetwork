import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

loss_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

TRAIN_SIZE = 5500
TEST_SIZE = 1000

x_train, y_train = (mnist.train.images[:TRAIN_SIZE, :], mnist.train.labels[:TRAIN_SIZE, :])
x_test, y_test = (mnist.test.images[:TRAIN_SIZE, :], mnist.test.labels[:TRAIN_SIZE, :])

LEARNING_RATE = 0.01
EPOCHS = 2000

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in range(EPOCHS):
		sess.run(training, feed_dict={x: x_train, y_: y_train})
		if i % 100 == 0:
			print("Epoch {}, Accuracy={}, Loss={}".format(i,
				sess.run(accuracy, feed_dict={x: x_test, y_: y_test}),
				sess.run(loss_function, feed_dict={x: x_train, y_:y_train})))
	# sess.run()

