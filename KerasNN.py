from keras.models import Sequential
from keras.layers import Dense

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
TRAIN_SIZE = 5500
TEST_SIZE = 1000
x_train, y_train = (mnist.train.images[:TRAIN_SIZE,:], mnist.train.labels[:TRAIN_SIZE,:])
x_test, y_test = (mnist.test.images[:TEST_SIZE,:], mnist.test.labels[:TEST_SIZE,:])

model = Sequential()

model.add(Dense(200, activation='relu', input_shape=(784,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='mse',
			  optimizer='sgd',
			  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

print(model.evaluate(x_test, y_test))