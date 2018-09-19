import pickle, gzip

def getData(fileName: str):
	with gzip.open(fileName, 'rb') as f:
		trainingD, validationD, testD = pickle.load(f, encoding='latin1')
	return (trainingD, validationD, testD)


if __name__ == '__main__':
	from pprint import pprint as pp
	trDat, valDat, teDat = getData("mnist.pkl.gz")
	print(trDat[0].shape)
	print(trDat[1].shape)
	