import tqdm
from collections import Counter
import numpy as np

#torch stuff:
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

#our packages
from shnetutil import dataset_base, projconfig, torchutils
from shnetutil.pipeline import loadMNIST, augmentation, batch

#from utils import batch

kDataLoader=False
kEpoch=False 		#use .epoch() generator interface
kSelfIter=True 		#use builder.__iter__()
kIterator=True 		#standalone iterator

MNIST_TRANSFORM = transforms.Compose((
	transforms.Pad(2),
	transforms.ToTensor(),
	transforms.Normalize((0.1,), (0.2752,))
))

def testDataLoader(mnist_train, bsize):
	train_loader = DataLoader(mnist_train, batch_size=bsize, shuffle=True, pin_memory=True)

	for i in range(epochs):
		trainiter = iter(train_loader)
		for b, batch_ in enumerate(trainiter):
			images, label = batch_
			print(f"[{i,b}], {images.shape}")

def test_epochgen(mnist_train, bsize):
	""" use .epoch() generator on the BatchBuilder """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()
		trainiter = trainbatchbuilder.epoch(False)

		for b, mybatch in enumerate(trainiter):
			#'mybatch' is an array of indices defining the minibatch samples
			#print(mybatch[10:])
			images, labels = batch.getBatchAsync(mnist_train, mybatch)
			#images, label = batch_
			print(f"[{i,b}]{mybatch.shape}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
		
def test_selfIter(mnist_train, bsize):
	""" use iter() on the BatchBuilder itself """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	labels2 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(trainbatchbuilder):		#trainbatchbuilder itself is iterable
			images, labels = mybatch
			print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels2.append(labels)
		print(labelcnt)
	return labels2	

def test_iterObj(mnist_train, bsize):
	""" standalone iterator .BatchIterator """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	train_loader = batch.BatchIterator(trainbatchbuilder) 	#create an iterator instance from our batch-builder

	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(train_loader):
			images, labels = mybatch
			print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1


if __name__ == '__main__':
	device = torchutils.onceInit(kCUDA = True)
	mnistdir = projconfig.getMNISTFolder()
	fashiondir = projconfig.getFashionMNISTFolder()

	#dataset = MNIST('mnist', train=True, download=True, transform=MNIST_TRANSFORM)
	mnist_train = loadMNIST.getdb(fashiondir, istrain=False, kTensor = False)
	print(f"mnist_train {len(mnist_train)} from {mnistdir}")

	#dbchunk = dataset_base.CustomDatasetSlice(mnist_train, (0,40))
	#mnist_train = dbchunk 	#use slice as our dataset

	MNIST_DATASET_SIZE = len(mnist_train)
	train_dataset_size = MNIST_DATASET_SIZE

	bsize = 1000
	epochs = 2

	#train_dataset, val_dataset = random_split(
	#	mnist_train, (train_dataset_size, MNIST_DATASET_SIZE - train_dataset_size)
	#)
	#print(f"train_dataset {len(train_dataset)}, val_dataset {len(val_dataset)}")

	if kDataLoader:
		testDataLoader(mnist_train, bsize)

	#
	# testing the 3 different ways to iterate a Bagging batch builder
	#

	#1: use our .epoch() generator 
	if kEpoch:	
		labels1 = test_epochgen(mnist_train, bsize)

	#2: use our iter(<Bagging>)
	if kSelfIter:	#use iter() on the BatchBuilder itself
		labels2 = test_selfIter(mnist_train, bsize)

	#2: use standalone iterator
	if kIterator:	#standalone iterator
		labels1 = test_iterObj(mnist_train, bsize)

	#print(labels1[0], labels2[0])	

	for i in range(len(labels1)):
		l1 = labels1[i]
		l2 = labels2[i]
		assert(np.equal(l1, l2).all())
	print(f"passed assert(np.equal(l1, l2).all())")	
		