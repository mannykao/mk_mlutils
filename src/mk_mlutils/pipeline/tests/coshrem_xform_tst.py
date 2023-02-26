# -*- coding: utf-8 -*-
"""

Title: COVID/pipeline/coshrem_xform_tst.py
    
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from skimage import data
from skimage.transform import resize

from shnetutil import coshrem_xform, projconfig, torchutils
from shnetutil.cplx import visual as cplxvisual
from shnetutil.pipeline import loadMNIST
from shnetutil.dataset import dataset_base

#
# 01/03/2022 Ported over from COVID/pipeline - mck.
#
def showimage(image, figsize=(6,6), show=False):
	plt.figure(figsize = figsize)
	plt.imshow(image, cmap = "gray")
	if show:
		plt.show()

def coin(size:int = 256):
	image = resize(data.coins(), (size,size))
	return image

def xform1image(coshxform, image:np.ndarray, doshow=False):
	torch_coeffs = coshxform.xform((image, -1))
	print(f"coeffs: {torch_coeffs.shape}, type: {type(torch_coeffs)}")

	# ### Visualizing the coefficients produced by torch
	coeffs_numpy = torch_coeffs.cpu().numpy()[0,:,:,:]
	print(type(coeffs_numpy), coeffs_numpy.shape, coeffs_numpy.dtype)

	j = 10
	coeff = coeffs_numpy[:,:,j]
	qUpper = np.percentile(np.absolute(coeff), 98)
	print(f"{qUpper=}")
	cplxvisual.complexImageShow(coeff/qUpper)
	
	if doshow:
		plt.show()

def xformbatch(coshxform, image, bsize=20):
	# ## Larger Batch to show advantage of using the pytorch only method
	image_complex = np.concatenate((image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 2)

	# Image batch done with repeating the same image 20 times
	batch_image = np.repeat(image[np.newaxis,:,:], bsize, axis=0)
	t = time.time()
	torch_coeffs = coshxform.batch_xform(batch_image);
	print(f"Elapsed time: torchsheardec2D() larger batch {time.time()-t:3f}ms")

def test_coin(device, doshow=False):
	image = coin(256)
	rows, cols = image.shape
	print(f"image: {image.shape}")
	#showimage(image, (6,6), True)

	#1: Generating the shearlet system for a 256x256 image
	coshxform = coshrem_xform.get_CoShXform(device, rows, cols, alpha=0.5)
	coshxform.start(device)
	shearlets, shearletIdxs, RMS, dualFrameWeights = coshxform.shearletSystem

	# The shearlets have 56 slices
	print(f"shearlets.shape {shearlets.shape}")

	#4: use pytorch 
	#shearlets_complex = coshxform.shearlets_complex
	#torch_shearlets	  = coshxform.torch_shearlets
	#device = coshxform.device

	#2: xform 1 image	
	xform1image(coshxform, image, doshow)

	#3: xform a batch of images
	xformbatch(coshxform, image, 20)


def compute_sparsity(coshxform, dbchunk, threshold=0.2, kLogging=False) -> tuple:
	total = 0
	below = 0
	percent = 98.0
	thresholds = np.ndarray(len(dbchunk))

	for i, entry in enumerate(dbchunk):
		img, label = entry
		image = resize(img, (32,32))
		torch_coeffs = coshxform.xform((image, -1))	#[1, 32, 32, 20]
		#print(torch_coeffs.shape, end='')
		coeffs_numpy = torch_coeffs.cpu().numpy()[0,:,:,:]	#(32, 32, 20)

		qUpper = np.percentile(coeffs_numpy, percent)
		thresholds[i] = qUpper.real

		nonzeros = coeffs_numpy > threshold
		n_non0 = np.count_nonzero(nonzeros)
		n_zeros = (coeffs_numpy.size - n_non0)
		below += n_zeros
		total += coeffs_numpy.size

		if kLogging:
			print(f"{qUpper=} ", end='')
			print(f"{n_zeros=}, {n_non0=}")

	adaptive = np.average(thresholds)
	print(f"adaptive threshold {adaptive}")
	
	return below, total

def xform_fashion(device):
	fashiondir = projconfig.getFashionMNISTFolder()
	fashion = loadMNIST.getdb(fashiondir, istrain=False, kTensor = False)
	print(f"fashion_test {len(fashion)} from {fashion}")
	dbchunk = dataset_base.CustomDatasetSlice(fashion, (0, len(fashion)))

	coshxform = coshrem_xform.get_CoShXform(device, 32, 32, alpha=0.5)

	#the 98 percentile threshold for test
	threshold = 0.2488514 	#train:  0.24853114677434868

	below, total = compute_sparsity(coshxform, dbchunk, threshold)
	print(f"{threshold=}: sparity ratio = {below/total}")	

	#t = time.time()
	#print(f"Elapsed time: xform_fashion() {time.time()-t:3f}s")


if __name__ == '__main__':
	device = torchutils.onceInit(kCUDA=True)

	#test_coin(device, doshow=False)

	xform_fashion(device)
