# -*- coding: utf-8 -*-
"""

Title: Unit test for batch iteration/batch sampling.
	
Created on Mon May 30 17:44:29 2023

@author: Manny Ko & Ujjawal.K.Panchal 

"""
import tqdm
from collections import Counter
import numpy as np

#torch stuff:
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

#our packages
from mk_mlutils import projconfig
from mk_mlutils.pipeline import augmentation, batch
from mk_mlutils.utils import torchutils
from datasets import dataset_base
from datasets.mnist import loadMNIST


kDataLoader=False
kBagging=True 			#test Bagging batch sampler
kBatchBuilder=True 		#test BatchBuilder batch sampler (standard minibatch)

MNIST_TRANSFORM = transforms.Compose((
	transforms.Pad(2),
	transforms.ToTensor(),
	transforms.Normalize((0.1,), (0.2752,))
))

def testDataLoader(mnist_train, bsize, epochs=1):
	train_loader = DataLoader(mnist_train, batch_size=bsize, shuffle=True, pin_memory=True)

	for i in range(epochs):
		trainiter = iter(train_loader)
		for b, batch_ in enumerate(trainiter):
			images, label = batch_
			print(f"[{i,b}], {images.shape}")

def test_epochgen(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.Bagging):
	""" use .epoch() generator on the BatchBuilder """
	print(f"test_epochgen({drop_last=}, {batchbuilder})")
	trainbatchbuilder = batchbuilder(mnist_train, bsize, drop_last=drop_last)
	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()
		trainiter = trainbatchbuilder.epoch(False)

		for b, mybatch in enumerate(trainiter):
			#'mybatch' is an array of indices defining the minibatch samples
			#print(mybatch[10:])
			images, labels = batch.getBatchAsync(mnist_train, mybatch)
			#images, label = batch_
			print(f" [{i,b}]{mybatch.shape}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
		
def test_selfIter(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.Bagging):
	""" use iter() on the BatchBuilder itself """
	print(f"test_selfIter({drop_last=}, {batchbuilder})")
	trainbatchbuilder = batchbuilder(mnist_train, bsize, drop_last=drop_last)
	labels2 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(trainbatchbuilder):		#trainbatchbuilder itself is iterable
			images, labels = mybatch
			print(f" [{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels2.append(labels)
		print(labelcnt)
	return labels2	

def test_iterObj(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.Bagging):
	""" standalone iterator .BatchIterator """
	print(f"test_iterObj({drop_last=}, {batchbuilder})")
	trainbatchbuilder = batchbuilder(mnist_train, bsize, drop_last=drop_last)
 	#create an iterator instance from our batch-builder
	train_loader = batch.BatchIterator(trainbatchbuilder)

	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(train_loader):
			images, labels = mybatch
			print(f" [{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1

def verifyLabels(labels1, labels2, tag='l1, l2'):
	for i in range(min(len(labels1), len(labels2))):
		l1 = labels1[i]
		l2 = labels2[i]
		print(f"[{i}]: ", end='')
		assert(np.equal(l1, l2).all())
	print(f"passed assert(np.equal({tag}).all())")	

def unitestBagging(
	dataset: dataset_base.DataSet, 
	bsize:int=128, 
	epochs:int=1,
	drop_last=False, 
	batchbuilder=batch.Bagging
):
	labels1 = test_epochgen(dataset, bsize=bsize, epochs=1, drop_last=drop_last, batchbuilder=batchbuilder)
	labels2 = test_selfIter(dataset, bsize=bsize, epochs=1, drop_last=drop_last, batchbuilder=batchbuilder)
	labels3 = test_iterObj (dataset, bsize=bsize, epochs=1, drop_last=drop_last, batchbuilder=batchbuilder)

	verifyLabels(labels1, labels2, tag='l1, l2')
	verifyLabels(labels1, labels3, tag='l1, l3')


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

	bsize = 992 	#test drop_last
	epochs = 1

	#train_dataset, val_dataset = random_split(
	#	mnist_train, (train_dataset_size, MNIST_DATASET_SIZE - train_dataset_size)
	#)
	#print(f"train_dataset {len(train_dataset)}, val_dataset {len(val_dataset)}")

	if kDataLoader:
		testDataLoader(mnist_train, bsize)
	#
	# testing the 3 different ways to iterate a Bagging batch builder
	#
	if kBagging:
		unitestBagging(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.Bagging)
		unitestBagging(mnist_train, bsize, epochs=1, drop_last=True, batchbuilder=batch.Bagging)
		print(" ")
	
	if kBatchBuilder:	# test iteration methods for BatchBuilder
		#1: use our .epoch() generator 
		labels11 = test_epochgen(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.BatchBuilder)
		labels21 = test_epochgen(mnist_train, bsize, epochs=1, drop_last=True, batchbuilder=batch.BatchBuilder)
		verifyLabels(labels11, labels21)

		#2: use our iter(<BatchBuilder>) - no supported
		#labels2 = test_selfIter(mnist_train, bsize, epochs=1, drop_last=True, batchbuilder=batch.BatchBuilder)

		#3: use standalone iterator
		labels1 = test_iterObj(mnist_train, bsize, epochs=1, drop_last=False, batchbuilder=batch.BatchBuilder)
		labels2 = test_iterObj(mnist_train, bsize, epochs=1, drop_last=True, batchbuilder=batch.BatchBuilder)
		verifyLabels(labels1, labels2)

		