"""
Title: Common Dispatcher file for all CVnn modules.

Description: The common dispatcher file is imported in all architecture files. it is used to recognize layers and weight operations for logging gradients.
*** this is a place holder file for the original cplx.dispacher.py to manage cplx dependencies

Created on Sun Mar 19 17:44:29 2023

@author: Ujjawal.K.Panchal & Manny Ko
"""
from typing import Union, Optional
import torch

#*** place holders  for the original cplx.dispacher.py to manage cplx dependencies

def getModelConfigStr(model):
	ourstr = ''
	if hasmethod(model, "getModelConfigStr"):
		ourstr = model.getModelConfigStr()
	elif hasmethod(model, "getConfigStr"):
		ourstr = model.getConfigStr()
	else:
		ourstr = type(model)
	return str(ourstr)

def get_histogram(tensor: torch.tensor, bins:int = 10):
	hist = None
	if torch.is_tensor(tensor):
		hist = torch.histc(tensor, bins = 10)
	else:
		hist = [torch.histc(tensor.real, bins = 10), torch.histc(tensor.imag, bins = 10)]
	return hist
	