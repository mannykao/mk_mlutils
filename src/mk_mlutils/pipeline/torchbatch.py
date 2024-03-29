# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import numpy as np
import torch
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

import mk_mlutils.projconfig

#enable cplx and CoShRem dependent code - mck
from mk_mlutils.projconfig import kUseCplx as kUseCplx

if kUseCplx:
	import cplxmodule

from mk_mlutils.pipeline import augmentation

from . import batch


def batch2device(device, imglist, labels, non_blocking=True):
	""" send imglist and labels to GPU """
	imglist = imglist.to(device, non_blocking=non_blocking)
	labels = labels.to(device, non_blocking=non_blocking)
	return imglist, labels

def torchify_batch(imglist, labels):
	"""
	turn imgs and labels to torch tensor if list of nparray.
	"""
	imgdispatch = {
		list: 		lambda imglist: torch.tensor(imglist),
		np.ndarray:	lambda imglist: torch.from_numpy(imglist),
		torch.Tensor: lambda imglist: imglist,
	}
	labdispatch = {
		list: 		lambda imglist: torch.tensor(imglist, dtype = torch.long),
		np.ndarray:	lambda imglist: torch.from_numpy(imglist),
		torch.Tensor: lambda imglist: imglist,
	}
	if kUseCplx:
		imgdispatch.update({cplxmodule.cplx.Cplx: lambda imglist: imglist})
		labdispatch.update({cplxmodule.cplx.Cplx: lambda lablist: lablist})

	imglist = imgdispatch[type(imglist)](imglist)
	labels  = labdispatch[type(labels)](labels)

	return imglist, labels

def getBatchAsync(
	device, 
	dbchunk, 
	batchindices, 
	imgXform:Callable = batch.NullXform(), 
	labelXform:Callable = batch.NullLabelXform(), 
	logging:bool=False
):
	""" Torch version of getBatchAsync() - transpose the imglist and sent to device """
	#1. get batch.
	imglist, labels = batch.getBatchAsync(dbchunk, batchindices, imgXform, labelXform, logging)

	#print(f"{labels.dtype=}")
	#assert(labels.dtype == np.int64)

	#2. make torch tensors if required.
	imglist, labels = torchify_batch(imglist, labels)
	
	#3. send to device
	imglist, labels = batch2device(device, imglist, labels)
	return imglist, labels

class ToTorchXform(augmentation.BaseXform):
	""" Null xform but convert to torch.Tensor
	---
	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		pass

	def __call__(self, sample:tuple):
		image, label = sample
		return torch.from_numpy(image), torch.from_numpy(label)
		

