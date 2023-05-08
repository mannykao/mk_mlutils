# -*- coding: utf-8 -*-
"""
Title: unit test for batch.py - 
	
Created on Tues Feb 28 16:01:29 2023

@author: Manny Ko 
"""
from collections import namedtuple, Counter
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Union, Optional
import numpy as np

from mk_mlutils import projconfig
import mk_mlutils.dataset.dataset_base as dataset_base
import mk_mlutils.dataset.datasetutils as datasetutils
import mk_mlutils.dataset.fashion as fashion

import mk_mlutils.pipeline.batch as batch
import mk_mlutils.pipeline.logutils as logutils

import mk_mlutils.utils.torchutils as torchutils


kRepoRoot="mk_mlutils/src/mk_mlutils"

def test_balancedSubset(validate=0.2) -> Tuple[dataset_base.DataSet, dataset_base.DataSet, dataset_base.DataSet]:
	#load_fashion() use datasetutils.getBalancedSubset() to subsample the test-set for validate
	train, test, validateset, *_ = fashion.load_fashion('train', validate=validate)
	print(f"{len(train)=}, {len(test)=}, {len(validateset)=}")
	return train, test, validateset

def test_epochgen(mnist_train, bsize, epochs=1, kLogging=False):
	""" use .epoch() generator on the BatchBuilder """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()
		trainiter = trainbatchbuilder.epoch(False)
		#trainiter = iter(train_loader)
		for b, mybatch in enumerate(trainiter):
			#'mybatch' is an array of indices defining the minibatch samples
			#print(mybatch[10:])
			images, labels = batch.getBatchAsync(mnist_train, mybatch)
			#images, label = batch_
			if kLogging: print(f"[{i,b}]{mybatch.shape}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
		
def test_selfIter(mnist_train, bsize, epochs=1, kLogging=False):
	""" use iter() on the BatchBuilder itself """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	labels2 = []
	for i in range(epochs):
		trainiter = iter(trainbatchbuilder)
		labelcnt = Counter()

		for b, mybatch in enumerate(trainiter):
			images, labels = mybatch
			if kLogging: print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels2.append(labels)
		print(labelcnt)
	return labels2	

def test_iterObj(mnist_train, bsize:int=256, epochs=1, kLogging=False):
	""" standalone iterator .BatchIterator """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	train_loader = batch.BatchIterator(trainbatchbuilder)

	labels1 = []
	for i in range(epochs):
		labelcnt = Counter()

		for b, mybatch in enumerate(train_loader):
			images, labels = mybatch
			if kLogging: print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1

def unitestBagging(dataset: dataset_base.DataSet, bsize:int=128, epochs:int=1):
	#batch.unitestBagging(validateset, bsize=256, epochs=1)
	labels1 = test_epochgen(dataset, bsize=256, epochs=1)
	labels2 = test_selfIter(dataset, bsize=256, epochs=1)
	labels3 = test_iterObj(dataset, bsize=256, epochs=1)

	for i in range(len(labels1)):
		l1 = labels1[i]
		l2 = labels2[i]
		l3 = labels3[i]
		assert(np.equal(l1, l2).all())
		assert(np.equal(l1, l3).all())
	print(f"passed assert(np.equal(l1, l2).all())")	
	print(f"passed assert(np.equal(l1, l3).all())")	


if __name__ == '__main__':
	torchutils.onceInit(kCUDA=True, cudadevice='cuda:1')

	projconfig.setRepoRoot(kRepoRoot, __file__)
	print(f"{projconfig.getDataFolder()=}")

	train, test, validateset = test_balancedSubset(validate=0.3)

	unitestBagging(validateset, bsize=512)
