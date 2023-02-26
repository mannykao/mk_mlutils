# -*- coding: utf-8 -*-
"""

Title: CIFAR10 loader

Created on Fri Oct 15 17:44:29 2021

@author: Manny Ko & Ujjawal.K.Panchal

"""
from pathlib import Path
import os.path as path
import numpy as np
#import torch
from PIL import Image
from skimage.color import (rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb, rgb2gray)
from torchvision import datasets
#from torchvision.utils import check_integrity

from shnetutil import projconfig
from shnetutil.dataset import dataset_base, datasetutils
from shnetutil.pipeline import augmentation
from shnetutil.pipeline.color import colorspace as colorspace

from . import BigFile, BigFileBuilder

import mkpyutils.dirutils as dirutils

cifar10BigDb = (projconfig.getCIFAR10Folder()/"cifar-test.dat", projconfig.getCIFAR10Folder()/"cifar-train.dat")

filelist = [
	"batches.meta",
	"data_batch_1",
	"data_batch_2",
	"data_batch_3",
	"data_batch_4",
	"data_batch_5",
	"test_batch",
]
def check_cifar(datasets_root: Path):
	cifardir = datasets_root/'cifar-10-batches-py'
	result = dirutils.direxist(cifardir)
	for ourfile in filelist:
		result &= path.exists(cifardir/ourfile)
	return result


def PIL2numpy(img: Image):
	return np.asarray(img)

def PIL2graynumpy(img: Image):
	""" return grayscale """
	return np.array(img.convert('L'))
	#lab = rgb2lab(np.asarray(img))
	#return lab[:, :, 0]

def PIL2numpyL(img: Image):
	""" return the L channel of Lab """
	lab = rgb2lab(np.asarray(img))
	return lab[:, :, 0]

def PIL2numpyLab(img: Image):
	lab = rgb2lab(np.asarray(img))
	return lab

def PIL2numpyYuv(img: Image):
	lab = rgb2yuv(np.asarray(img))
	return lab

#full dataset xforms for BigFiling.
#Suggestion: Move to pipeline.augmentation
def grayXform(dataset, n_jobs):
	"""rgb2gray xform for BigFileBuilder"""
	images = np.asarray(dataset_base.getCoeffs(dataset))
	images = rgb2gray(images)
	return images.astype(np.single)

def LabXform(dataset, n_jobs):
	""" rgb2lab xform for BigFileBuilder """
	images = np.asarray(dataset_base.getCoeffs(dataset))
	images = rgb2lab(images)
	img0 = images[0]
	print(img0.shape, img0.dtype)
	return images.astype(np.single)

def LXform(dataset, n_jobs):
	lab_images = LabXform(dataset, n_jobs)
	l_images = lab_images[...,0]
	return l_images

def checkBigFiles(testfile, trainfile):
	return path.exists(testfile) and path.exists(trainfile)

def cifar2BigFile(
	dataset, 
	outputfile:str, 
	xform=LabXform, 
	kZip=False,
	bigfileclass = "Big1Chunk",
	colorspace = "lab"
) -> BigFile:
	""" Build a BigFile out of 'dataset' apply 'xform' """
	builder = BigFileBuilder.BigFileBuilder(
		outputfile,
		bigfileclass=bigfileclass,
		xform=xform,	#iterate 'dataset' to convert it to an array of coefficients
		n_jobs=4
	)
	builder.doit(dataset, verify=True, kPickle=False, kZip=kZip, colorspace=colorspace)
	
	#2: open the completed BigChunk for read to make sure it is working
	bigfile = builder.targetclass(outputfile, kPickle=False)	#BigFile.Big1Chunk
	return bigfile

#rgb -> colorspaces

color_dispatch = {
	"rgb": PIL2numpy,
	"lab": PIL2numpyLab,
	"yuv": PIL2numpyYuv,
	"grayscale": PIL2graynumpy,
	"lum_lab":   PIL2numpyL,
#	"xyz": "rgb2xyz", etc. etc.
}

dsetXform_dispatch = {
	"lab": LabXform,
	"lum_lab": LXform,
	"grayscale": grayXform,
}


def err(type_):
	raise NotImplementedError(f'Color space conversion {type_} not implemented yet')

def download_cifar(
	datasets_root: Path = None,
	colorspace = "grayscale",
) -> tuple:

	"""
	About:
	---
	Args:
		1. dataset_root
		2. colorspace.
	---
	Issue <!>: This currently assumes that we only use grayscale to deal with CIFAR. We need to refactor to make provision for color.
	"""
	datasets_root = projconfig.getCIFAR10Folder()
	check_cifar(datasets_root) #TODO: redundant call remove.
	#print(datasets_root)
	
	cifar_missing = not check_cifar(datasets_root)

	#1: select conversion to grayscale method or keep rgb color:
	colorxform = color_dispatch.get(colorspace, lambda x: np.asarray(x)) #tograyscale.get(colorspace, lambda x: np.asarray(x))
	cifar10_train = datasets.CIFAR10(datasets_root, train=True, download=cifar_missing, transform = colorxform)
	cifar10_test  = datasets.CIFAR10(datasets_root, train=False, download=cifar_missing, transform = colorxform)
	return cifar10_train, cifar10_test


if (__name__ == '__main__'):
	datasets_root = projconfig.getCIFAR10Folder()
	training_set, test_set = download_cifar(datasets_root, colorspace="grayscale")
	print(f"training_set.colorspace {training_set.colorspace}")
