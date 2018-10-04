from Network import Network
from Data import *

def main():
	#	Each of these data containers looks like ([X,784]: data, [X]: label) where data \in (0,1)
	trDat, valDat, teDat = getData("mnist.pkl.gz")

	n = Network(784, 10, [100])

	# print([i.shape for i in n.weights])

	n.Train(trDat, 10, True, teDat)


if __name__ == '__main__':
	main()
