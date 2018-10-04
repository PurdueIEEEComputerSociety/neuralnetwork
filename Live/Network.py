import numpy as np

EPSILON = 0.001

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
	return sigmoid(z)*(1-sigmoid(z))

class Network:
	def __init__(self, inputSize, outputSize,
	 hiddenLayers, activationFunction=sigmoid, activationDerivative=sigmoidPrime):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.activationFunction = activationFunction
		self.activationDerivative = activationDerivative

		self.weights = [np.random.rand(i, j) - 0.5 for i,j in zip(list([inputSize] + hiddenLayers),
																			list(hiddenLayers + [outputSize]))]
	
	def BackwardPass(self, inputt, label):

		pVec = np.array([0] * self.outputSize)
		pVec[label] = 1

		outputs = [inputt]
		Z = []
		for ind, w in enumerate(self.weights):
			z = np.matmul(outputs[ind], w)
			Z.append(z)
			outputs.append(self.activationFunction(z))

		derWeights = [np.zeros(w.shape) for w in self.weights]

		for i in range(len(outputs)-1, 0, -1):
			if (i == len(outputs)-1):
				delta = 2*(outputs[i] - pVec) * self.activationDerivative(Z[-1])
			else:
				delta = self.weights[i].dot(delta) * self.activationDerivative(Z[i-1])

			derWeights[i-1] = np.outer(outputs[i-1], delta)

		return derWeights


	def ForwardPass(self, inputt):
		outputs = [inputt]

		for ind, w in enumerate(self.weights):
			z = np.matmul(outputs[ind], w)
			outputs.append(self.activationFunction(z))

		return np.argmax(outputs[-1])

	def Test(self, data, classifications):
		successes = 0
		for i in range(data.shape[0]):
			guess = self.ForwardPass(data[i])
			if (guess == classifications[i]):
				successes += 1
		return successes/data.shape[0]

	def Train(self, inputt, epochs, verbose: False, testDat: None):
		inputSize = inputt[0].shape[0]
		batchSize = inputSize//100

		for i in range(epochs):
			if verbose:
				print("Epoch {}: Accuracy {}".format(i, self.Test(*testDat)))

			newOrder = np.random.permutation(inputSize)
			inputt = (inputt[0][newOrder], inputt[1][newOrder])

			batches = [range(i, i+batchSize) for i in range(0, inputSize, batchSize)]
			for batch in batches:
				accDerWeights = [np.zeros(w.shape) for w in self.weights]
				for b in batch:
					dW = self.BackwardPass(inputt[0][b], inputt[1][b])
					accDerWeights = [sum(x) for x in zip(accDerWeights, dW)]
				self.weights = [w - EPSILON * dW for w,dW in zip(self.weights, accDerWeights)]



if __name__ == '__main__':
	from Data import getData
	trDat, valDat, testD = getData("../mnist.pkl.gz")
	n = Network(784, 10, [75])
	for i in n.weights:
		print(i.shape)
	# print(n.ForwardPass(trDat[0][0]))
	# print(n.Test(testD[0], testD[1]))
	# print(trDat[1][0])
	# for i in range(20):
	# 	print(n.Test(*testD))
	# 	n.weights = [w - dW for w,dW in zip(n.weights, n.BackwardPass(trDat[0][i], trDat[1][i]))]
	n.Train(trDat, 3, True, testD)


