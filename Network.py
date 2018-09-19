import numpy as np

EPSILON=0.001
RELU_LEAKAGE=100


def LeakyReLU(z: list):
	"""
	Activation function (mutates x)
	"""
	z[z<0]/=RELU_LEAKAGE

def LeakyReLUPrime(x: list):
	"""
	Derivative of func
	"""
	if x < 0:
		return 1/RELU_LEAKAGE
	else:
		return 1


class Network:
	def __init__(self, inputSize: int, outputSize: int, hiddenLayers: list, activationFunction = LeakyReLU, activationDerivative = LeakyReLUPrime):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.activationFunction = activationFunction
		self.activationDerivative = activationDerivative

		#	To ease training, the bias will be accounted for by an additional weight that will have a constant value of 1 for each layer

		#	Weights go from one layer to the next (each neuron has a weight for every input (i.e. neuron) in the prior layer)
		self.weights = np.array([np.random.rand(i, j) - 0.5 for i,j in zip([i+1 for i in list([inputSize] + hiddenLayers)],	#	Input (plus 1 for the bias)
																		   list(hiddenLayers + [outputSize]))])				#	Output

	def ForwardPass(self, inputt: list) -> int:
		"""
		Classify an input
		"""
		outputs = [inputt]	#	Input info passes straight through

		#	For each layer, compute the outputs using the previous inputs
		for ind, w in enumerate(self.weights):
			z = np.matmul(np.append(outputs[ind],1), w)	#	z = W*input + bias
			self.activationFunction(z)					#	output = activation(z)
			outputs.append(z)
		
		return np.argmax(outputs[-1])	#	Return the index of the largest value prediction (corresponds to which digit is predicted)

	def BackwardPass(self, inputt: list, outputt: int):
		"""
		Train the network (backprop)
		"""
		#	Convert the output from an int to a probability vector
		pVec = [0] * len(self.outputSize)
		pVec[output] = 1

		#	Perform a forward pass while maintaining all state
		outputs = [inputt]
		Z = []
		for ind, w in enumerate(self.weights):
			z = np.matmul(np.append(outputs[ind],1), w)	#	z = W*input + bias
			Z.append(z.copy())							#	retain z values
			self.activationFunction(z)					#	output = activation(z)
			outputs.append(z)
		
		outputs[-1]/=outputs[-1].max()	#	Normalize the output vector

		err = 0.5*(pVec - outputs[-1])**2	#	Total output error
		err = err.sum()	#	sum of squared error


		# dErr/dw = dErr/dOut * dOut/dNet (Activation prime) * dNet/dw (Output)
		# w -= learn * dErr/dw


	def Train(self, inputt: list, epochs: int, learnRate: float, verbose: bool):
		"""
		Train the network (stochastic gradient descent)
		"""
		#NOTE: Remember to shuffle
		pass


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
	n = Network(784, 10, [75, 50, 50])
	# for i in n.weights:
	# 	print(i.shape)
	print(n.ForwardPass(np.random.rand(784)))




