import time
from collections import Counter
from operator import itemgetter
from pathlib import Path
# -*- coding: utf-8 -*-
"""
Title: Model Factory - 
	
Created on Fri Aug 21 16:01:29 2021

@author: Manny Ko & Ujjawal.K.Panchal
"""
from typing import Union, Optional, Tuple, Iterable
import matplotlib.pyplot as plt

import numpy as np
import torch


def singularValues(v:Union[torch.Tensor, np.ndarray], klogging=False) -> np.ndarray:
	""" return singular Values for 'v' as a ndarray """
	if type(v) == torch.Tensor:
		v = v.detach().cpu().to(torch.float32) 	#in case it is quantized floats (e.g. BFloat16)
	S = torch.linalg.svdvals(v)
	values = S.numpy()

	return values

def binFloats(v:np.ndarray, nbins:int=12, lowerB:float=0.) -> Tuple[list, tuple]:
	""" Return a set of bins for [lowerB..upperB(v)] """
	min, max = np.min(v), np.max(v)
	delta = (max - lowerB)/nbins
	bins = [i*delta for i in range(0, nbins)]
	bins[-1] = max
	return bins, (lowerB, max)

def floatStats(v:Union[list, np.ndarray], bins:Iterable, tag:str="Singular Values"):
	""" Format our floating bins and dump it """
	binfmt = ["%.5f" % elem for elem in bins]
	print(f"{tag}(bins{binfmt}) [{0}:{v.max()}]")
	c = Counter(v)
	c = sorted(c.items(), key=itemgetter(0))

def histogramCounts(histogram:Union[list, np.ndarray]) -> list:
	labelcnt = Counter(histogram)
	return labelcnt

def dumpCounter(cnt:Counter, nbins:int=12, tag:str=" "):
	""" Dump counter 'cnt' forcing empty bins to show 0 counts """
	zeros = Counter(range(nbins))
	labelcnt = cnt + zeros	#create empty bins to show 0 counts
	c = sorted(labelcnt.items(), key=itemgetter(0))
	c = [(k, v-1) for k, v in c]
	print(f"{tag}{c}")	

def verifyBin(S:np.ndarray, bins:list, b:int=1, labelcnt:list=[]) -> bool:
	count2 = 0
	for v in S:
		if (v >= bins[b-1]) and (v < bins[b]):
			count2 += 1
	print(f"bin[{bins[b-1]:.5f}:{bins[b]:.5f}]: {count2=} {labelcnt[b]=}")		
	return (count2 == labelcnt[b])

def singularStats(w:torch.Tensor, eigen:bool=True, nbins:int=12, name:str="", klogging:bool=False):
	print(f"singularStats({name}) {w.shape} eigen={eigen} ->")

	if len(w.shape) >= 2:
		S = singularValues(w)

		if eigen:
			eigenV = np.sqrt(S)
			S = eigenV
			tag = " eigen values"
		else:
			tag = " singular values"

		bins, frange = binFloats(S, nbins)

		histogram = np.digitize(S, bins, right=True) 
		#print(f"{histogram}, {histogram.min()}, {histogram.max()}")
		floatStats(histogram, bins, tag=tag)

		labelcnt = histogramCounts(histogram)

		if False:	#manual counting 1 bin for checking
			verifyBin(S, bins, 1, labelcnt)
			verifyBin(S, bins, 2, labelcnt)
			verifyBin(S, bins, 3, labelcnt)

		dumpCounter(labelcnt, nbins, tag=" histogram")
	else:
		print(f"{name} has less than 2d")