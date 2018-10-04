import pickle, gzip

def getData(fileName):
	with gzip.open(fileName, 'rb') as f:
		trDat, valDat, testD = pickle.load(f, encoding='latin1')
	return (trDat, valDat, testD)

if __name__ == '__main__':
	trDat, valDat, testD = getData("../mnist.pkl.gz")
	print(type(trDat))
	print(trDat[0].shape)
	print(trDat[1].shape)