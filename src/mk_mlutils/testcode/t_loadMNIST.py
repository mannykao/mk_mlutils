# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

from mk_mlutils import projconfig
from datasets.mnist import mnist

#
# testing mnist.getdb() and  in standalone mode
#
class NullXform(object):
	""" Null xform 
	---import shnetutil.cplx as shcplx

	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		print(f"NullXform({kwargs})")
		self.kwargs = kwargs
		pass

	def __call__(self, sample):
		print(f"__call__")
		return sample

def getBatch(
	dataset:List, 
	imgXform:Callable = NullXform() 
):
	xformed = nullxform(dataset)
	print(f"getBatch({imgXform}) ")
	print([sample.label for sample in xformed])


if __name__ == '__main__':
	fashiondir = projconfig.getFashionMNISTFolder()

	#dataset = MNIST('mnist', train=True, download=True, transform=MNIST_TRANSFORM)
	mnist_train = mnist.getdb(fashiondir, istrain=False, kTensor = False)
	print(f"mnist_train {len(mnist_train)} from {fashiondir}")

	test = []
	for i in range(10):
		sample = mnist_train[i]
		#print(sample.label)
		test.append(sample)
	print([sample.label for sample in test])

	nullxform = NullXform()
	testxformed = nullxform(test)
	print(len(testxformed))

	getBatch(test)