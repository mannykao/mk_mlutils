# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import numpy as np
import torch
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

from . import batch

class NullXform(object):
	""" Null xform 
	---
	Args: (N/A).

	"""
	def __init__(self):
		pass

	def __call__(self, sample):
		return sample
		
def batch2device(device, imglist, labels, non_blocking=True):
	""" send imglist and labels to GPU """
	imglist = imglist.to(device, non_blocking=non_blocking)
	labels = labels.to(device, non_blocking=non_blocking)
	return imglist, labels

def getBatchAsync(
	device, 
	dbchunk, 
	batchindices, 
	imgXform:Callable = batch.NullXform(), 
	labelXform:Callable = batch.NullXform(), 
	logging:bool=False
):
	""" Torch version of getBatchAsync() - transpose the imglist and sent to device """
	#1. get batch.
	imglist, labels = batch.getBatchAsync(dbchunk, batchindices, imgXform, labelXform, logging)

	imglist = imglist
	labels = torch.tensor(labels) #EDIT: dtype brought out.
	#3. send to device
	imglist, labels = batch2device(device, imglist, labels)
	return imglist, labels

