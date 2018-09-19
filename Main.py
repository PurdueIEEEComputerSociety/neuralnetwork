from Network import Network
from Data import *

def main():
	#	Each of these data containers looks like ([X,784]: data, [X]: label) where data \in (0,1)
	trDat, valDat, teDat = getData("mnist.pkl.gz")

	n = Network(784, 10, [75, 50, 50])
	n.Test(*teDat)


if __name__ == '__main__':
	main()
