# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import numpy as np
import torch
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

from mk_mlutils.pipeline import augmentation

from . import batch

class ToTorchXform(augmentation.Base):
	""" Null xform 
	---
	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		pass

	def __call__(self, sample):
		return torch.Tensor(sample)
		
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

	print(f"{type(imglist)=}")
	imglist = torch.tensor(imglist)
	labels = torch.tensor(labels) #EDIT: dtype brought out.
	#3. send to device
	imglist, labels = batch2device(device, imglist, labels)
	return imglist, labels

