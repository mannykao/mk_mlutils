
# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Sun Aug 4 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
from typing import List, Tuple, Optional, Callable
import numpy as np
import abc

from mk_mlutils import projconfig 


class BaseXform(metaclass=abc.ABCMeta):		#TODO: rename to BaseXform
	""" Abstract baseclass for augmentation xforms 
	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		pass

	def __call__(self, sample):
		return sample

	def rewind(self):
		pass	

class NullXform(BaseXform):
	""" Null xform 
	---import mk_mlutils.cplx as shcplx

	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		pass

	def __call__(self, sample):
		return sample

