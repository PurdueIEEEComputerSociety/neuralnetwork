import numpy as np

EPSILON=0.001
RELU_LEAKAGE=100

def LeakyReLU(z: np.ndarray):
	"""
	Activation function
	"""
	return np.where(z > 0, z, z/RELU_LEAKAGE)

def LeakyReLUPrime(z: np.ndarray):
	"""
	Derivative of activation function
	"""
	return np.where(z > 0, 1, 1/RELU_LEAKAGE)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network:
	def __init__(self, inputSize: int, outputSize: int, hiddenLayers: list, activationFunction = sigmoid, activationDerivative = sigmoid_prime):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.activationFunction = activationFunction
		self.activationDerivative = activationDerivative

		#	To ease training, the bias will be accounted for by an additional weight that will have a constant value of 1 for each layer

		#	Weights go from one layer to the next (each neuron has a weight for every input (i.e. neuron) in the prior layer)
		# self.weights = np.array([np.random.rand(i, j) - 0.5 for i,j in zip([i+1 for i in list([inputSize] + hiddenLayers)],	#	Input (plus 1 for the bias)
		# 																   list(hiddenLayers + [outputSize]))])				#	Output
		self.weights = [np.random.rand(i, j) - 0.5 for i, j in
								 zip(list([inputSize] + hiddenLayers),  #	Input (plus 1 for the bias)
									 list(hiddenLayers + [outputSize]))]  #	Output

	def ForwardPass(self, inputt: list) -> int:
		"""
		Classify an input
		"""
		outputs = [inputt]	#	Input info passes straight through

		#	For each layer, compute the outputs using the previous inputs
		for ind, w in enumerate(self.weights):
			z = np.matmul(outputs[ind], w)	#	z = W*input + bias
			outputs.append(self.activationFunction(z))	#	output = activation(z)

		return np.argmax(outputs[-1])	#	Return the index of the largest value prediction (corresponds to which digit is predicted)

	def BackwardPass(self, inputt: list, outputt: int):
		"""
		Train the network (backprop)
		"""
		#	Convert the output from an int to a probability vector
		pVec = np.array([0] * self.outputSize)
		pVec[outputt] = 1

		#	Perform a forward pass while maintaining Z state
		outputs = [inputt]
		Z = []
		for ind, w in enumerate(self.weights):
			z = np.matmul(outputs[ind], w)	#	z = W*input + bias
			Z.append(z)							#	retain z values
			outputs.append(self.activationFunction(z))	#	output = activation(z)

		outputs[-1] /= outputs[-1].max()	#	Normalize the output vector

		derWeights = [np.zeros(w.shape) for w in self.weights]

		#	Minimizing sum of squared error
		for i in range(len(outputs)-1, 0, -1):	#	Iterate backwards over the outpus until you get to the 0th (0th is input, can't change)
			#	For the final output, calculate the derivative using the cost function
			if (i == len(outputs)-1):
				delta = 2*(outputs[i] - pVec) * self.activationDerivative(Z[-1])	#	Derivative of cost w/ respect to output
			else:
				#	Take the next layer's weights matrix by derivatives vector
				#	Sum them because each neuron in the prior layer is going to affect multiple neurons in the next layer.
				#	The sum is summing the effect for each neuron in the next layer (i.e. if half the neurons in the next layer say
				#	this layer's neuron should change 1 way, and the other half say the other way, don't move)
				# delta = np.matmul(self.weights[i], delta).sum() * self.activationDerivative(Z[i-1])
				# delta = self.activationDerivative(Z[i - 1]) * np.dot(self.weights[i], delta)
				# print(self.weights[i].shape, self.weights[i].T.shape)
				# print(delta.shape, self.activationDerivative(Z[i-1]).shape, self.weights[i].dot(delta).shape)
				delta = self.weights[i].dot(delta) * self.activationDerivative(Z[i-1])

			derWeights[i-1] = np.outer(outputs[i-1],	#	Prior layer's activation (with an additional 1 for the bias)
									 delta)
			# derWeights[i - 1] = np.outer(delta, outputs[i - 1])

		#	Return matrix of weight adjustments
		return derWeights

	def Train(self, inputt: list, epochs: int, verbose: bool = False, trainDat: list = None):
		"""
		Train the network (stochastic gradient descent)
		"""
		inputSize = inputt[0].shape[0]
		batchSize = inputSize//100
		for i in range(epochs):
			if verbose:
				print("Epoch {0}: Accuracy {1}".format(i, self.Test(*trainDat)))

			#	Shuffle the data
			newOrder = np.random.permutation(inputSize)
			inputt = (inputt[0][newOrder], inputt[1][newOrder])

			batches = [range(i, i+batchSize) for i in range(0, inputSize, batchSize)]
			for batch in batches:
				accDerWeights = [np.zeros(w.shape) for w in self.weights]
				for b in batch:
					dW = self.BackwardPass(inputt[0][b], inputt[1][b])
					accDerWeights = [sum(x) for x in zip(accDerWeights, dW)]

				#	Update the weights
				self.weights = [w - EPSILON * dw for w, dw in zip(self.weights, accDerWeights)]

	def Test(self, data: list, classifications: list) -> float:
		"""
		Test the network accuracy on the data. Returns the fraction
		"""
		successes = 0
		for i in range(data.shape[0]):
			guess = self.ForwardPass(data[i])
			if (guess == classifications[i]):
				successes += 1
		return successes/data.shape[0]

if __name__ == '__main__':
	n = Network(784, 10, [75])
	# for i in n.weights:
	# 	print(i.shape)
	# print(n.ForwardPass(np.random.rand(784)))
	# a = n.BackwardPass(np.random.rand(784), 2)

	# print([i.shape for i in a])
	# print([i.shape for i in n.weights])


