# -*- coding: utf-8 -*-
"""
Title: unit test for batch.py support for torch.DataLoader - 
	
Created on Tues Feb 28 16:01:29 2023

@author: Manny Ko 
"""

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.mnist import mnist
from datasets.utils.xforms import GreyToFloat
from mk_mlutils.pipeline import batch, torchbatch

batchsize=100

kDataLoader=False 	#torch code we want to emulate
kBatchBuilder=True
kBagging=True


if __name__ == '__main__':
	#torchutils.onceInit(kCUDA=True, cudadevice='cuda:1')
	seqmnist = mnist.MNIST(split="test")

	if kDataLoader:
		dl_val   = DataLoader(seqmnist, batch_size = batchsize, shuffle = False, num_workers = 1, pin_memory = True)

		mylabels1 = []
		for mybatch, labels in tqdm(dl_val):
			mylabels1.append(labels)
		print(len(mylabels1))
	
	if kBatchBuilder:
		trainbatchbuilder = batch.BatchIterator(batch.BatchBuilder(seqmnist, batchsize, shuffle = False))

		mylabels2, b = [], 0
		for mybatch, labels in tqdm(trainbatchbuilder):
			mylabels2.append(labels)
			if (b == 0):
				pass
				#print(type(mybatch[0]), type(mybatch[1]))
				#print(batch[0], batch[1])
			b += 1

	if kBagging:		
		trainbatchbuilder = batch.Bagging(seqmnist, batchsize=batchsize)

		mylabels3 = []
		for mybatch, labels in tqdm(trainbatchbuilder):
			mylabels3.append(labels)
