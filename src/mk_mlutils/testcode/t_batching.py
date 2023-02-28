# -*- coding: utf-8 -*-
"""
Title: Training pipeline utils - 
	
Created on Wed Sept 1 16:01:29 2020

@author: Manny Ko & Ujjawal.K.Panchal
"""
import re
from collections import namedtuple, Counter
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Union, Optional

from mk_mlutils import projconfig
import mk_mlutils.dataset.dataset_base as dataset_base
import mk_mlutils.dataset.datasetutils as datasetutils
import mk_mlutils.dataset.fashion as fashion

import mk_mlutils.pipeline.batch as batch
import mk_mlutils.pipeline.logutils as logutils


#from ..pipeline import loadMNIST, augmentation, dbaugmentations, trainutils

kRepoRoot="mk_mlutils/src/mk_mlutils"

def test_balancedSubset(validate=0.2) -> Tuple[dataset_base.DataSet, dataset_base.DataSet, dataset_base.DataSet]:
	#load_fashion() use datasetutils.getBalancedSubset() to subsample the test-set for validate
	train, test, validateset, *_ = fashion.load_fashion('train', validate=validate)
	print(f"{len(train)=}, {len(test)=}, {len(validateset)=}")
	return train, test, validateset

def test_epochgen(mnist_train, bsize, epochs=1):
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
			print(f"[{i,b}]{mybatch.shape}, {images.shape}")
			labelcnt.update(labels)
			labels1.append(labels)
		print(labelcnt)	
	return labels1
		
def test_selfIter(mnist_train, bsize, epochs=1):
	""" use iter() on the BatchBuilder itself """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	labels2 = []
	for i in range(epochs):
		trainiter = iter(trainbatchbuilder)
		labelcnt = Counter()

		for b, mybatch in enumerate(trainiter):
			images, labels = mybatch
			print(f"[{i,b}]{type(mybatch)}, {images.shape}")
			labelcnt.update(labels)
			labels2.append(labels)
		print(labelcnt)
	return labels2	

def test_iterObj(mnist_train, bsize:int=256, epochs=1):
	""" standalone iterator .BatchIterator """
	trainbatchbuilder = batch.Bagging(mnist_train, bsize)
	train_loader = batch.BatchIterator(trainbatchbuilder)

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
	projconfig.setRepoRoot(kRepoRoot, __file__)
	print(f"{projconfig.getDataFolder()=}")

	train, test, validateset = test_balancedSubset(validate=0.3)

	test_epochgen(validateset, bsize=256, epochs=1)
	test_selfIter(validateset, bsize=256, epochs=1)
	test_iterObj(validateset, bsize=256, epochs=1)

