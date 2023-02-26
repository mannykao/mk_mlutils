# -*- coding: utf-8 -*-
"""

Title: cifar10 support.
    
Created on Thurs July 6 17:44:29 2020

@author: Manny Ko

"""
from collections import namedtuple
from typing import List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
#import torch
from PIL import Image
from torchvision import datasets

from mkpyutils import dirutils
from shnetutil import projconfig
from shnetutil.dataset import dataset_base, datasetutils
from shnetutil.pipeline import augmentation, BigFile, loadCIFAR, dbaugmentations, trainutils

kBigFiles = ["cifar-test.dat", "cifar-train.dat"]


def getlabels(dataset):
	dispatch = {
		#grab the whole ndarray of labels
		datasets.cifar.CIFAR10: lambda dataset: dataset.targets, 	#torchvision: for speed get this handle
		CustomCIFARDB: lambda db: db.labels,
		BigFile.BigChunk: lambda bigfile: bigfile.labels,
		BigFile.Big1Chunk: lambda bigfile: bigfile.labels,
	}
	#print(f"getlabels: {type(dataset)}")
	getfunc = dispatch.get(type(dataset), None)
	if getfunc: 
		labels = getfunc(dataset)
	else:
		labels = [label for img, label in dataset]	
	return labels

def get_stdmean(customdb) -> tuple:
	if customdb.isGrayScale():
		mean, std = customdb.getstats()
		stats = mean, std
#		print(f"{type(customdb)=} {stats=} {mean=} {std=}")
	else:
		imgs = np.asarray(dataset_base.getCoeffs(customdb))
		#print(f"imgs.shape {imgs.shape}")
		mean = (np.mean(imgs[:,:,:,0])/255, np.mean(imgs[:,:,:,1])/255, np.mean(imgs[:,:,:,2])/255)
		std  = (np.std(imgs[:,:,:,0])/255, np.std(imgs[:,:,:,1])/255, np.std(imgs[:,:,:,2])/255)
		stats = mean, std
	return stats	

def colorspace_dispatch(dataset):
	#print(f"colorspace_dispatch {type(dataset)}")
	colorspace = "rgb" 	#assume it is the original colorspace cifar is in
	if issubclass(type(dataset), dataset_base.DataSet):
		colorspace = dataset.colorspace
	return colorspace

class CustomCIFARDB(dataset_base.DataSet):
	kGrayScaleStats = (0.4808612549785551, 0.23919067571925343)
	#precomputed stats for the training set:
	kStats = {
		"grayscale": kGrayScaleStats,
		"lum_lab":	 (0.19942640667130146, 0.0951312530961822), 	#L of Lab
		"lab":		 ((0.19942643408681832, 0.0015520177635492062, 0.022452247844022864), (0.09513126448089002, 0.039810808967141544, 0.06306255564970129)),
	}
	kGrayScales = set(["grayscale", "lum_lab"])

	def __init__(self, 
		dataset: Union[datasets.cifar.CIFAR10, BigFile.BigChunk],
		colorspace:str ="grayscale"		#output colorspace - i.e. what is returned by [] 
	):
		super().__init__(name="CIFAR10", colorspace=colorspace)

		#assert(issubclass(type(dataset), datasets.cifar.CIFAR10))
		self.dataset = dataset
		self.inputcolorspace = colorspace_dispatch(dataset)
 		#grab the whole ndarray of images (50000, 32, 32, 3). Note: it is the original pixels without transform
#		self.images = dataset.data
		self.labels = getlabels(dataset)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index) -> dataset_base.ImageDesc:
		entry = self.dataset[index]
		return dataset_base.ImageDesc(*entry)

	def isGrayScale(self):
		return self.colorspace in CustomCIFARDB.kGrayScales

	@classmethod
	def colorspaces(cls):
		return cls.kStats.keys()

	def getstats(self, recompute=False):
		stats = CustomCIFARDB.kStats.get(self.colorspace)
		if (stats is None) or recompute:
			stats = get_stdmean(self)
		return stats	

def getcifarstats(colorspace:str) -> tuple:
	return CustomCIFARDB.kStats.get(colorspace)

def download_cifar(
	datasets_root: Path,
	colorspace: str = "grayscale"
) -> Tuple[CustomCIFARDB]:
	cifar10_train, cifar10_test = loadCIFAR.download_cifar(datasets_root, colorspace=colorspace)
	cifar10_train = CustomCIFARDB(cifar10_train, colorspace=colorspace)
	cifar10_test = CustomCIFARDB(cifar10_test, colorspace=colorspace)
	return cifar10_train, cifar10_test

def make_bigfiles(test_set, training_set, testfile, trainfile, regen=False, colorspace = "lab"):
	filesok = loadCIFAR.checkBigFiles(testfile, trainfile)
	if regen or (not filesok):
		print(f"make_bigfiles->")
		filesok  = loadCIFAR.cifar2BigFile(
									test_set, testfile,
									xform=loadCIFAR.dsetXform_dispatch[colorspace],
									bigfileclass="Big1Chunk",
									colorspace = colorspace,
							) != None
		filesok &= loadCIFAR.cifar2BigFile(
									training_set, trainfile,
									xform=loadCIFAR.dsetXform_dispatch[colorspace],
									bigfileclass="Big1Chunk",
									colorspace = colorspace,
							)	!= None
	return filesok

def colorSmartBigFiles(trainfile: str, testfile: str, colorspace: str, rgb_training_set, rgb_test_set):
	#print("===(colorsmarts)===")
	try:
		cifar10_test  = BigFile.Big1Chunk(testfile, colorspace = colorspace)
		#print("cifar10_test loaded in buffer successfully.")
		#print(f"{cifar10_test.colorspace=}; {colorspace=}.")
		assert(cifar10_test.colorspace == colorspace), f"\
		(saved file colorspace) {cifar10_test.colorspace}!= {colorspace} (given colorspace). Hence, Remaking bigfiles."
	except Exception as e:
		print(f"encountered exception: {e}")

		result = make_bigfiles(rgb_test_set, rgb_training_set,
							   testfile, trainfile,
							   regen = True,
							   colorspace = colorspace,
				)
	cifar10_test = BigFile.Big1Chunk(testfile, colorspace = colorspace) #TODO: Generalize!
	cifar10_train = BigFile.Big1Chunk(trainfile, colorspace = colorspace) #TODO: Generalize!
	#print("===(/colorsmarts)===")
	return cifar10_train, cifar10_test

def load_cifar(
	trset: str,			#train|test for training
	validate: float = 0.2,		#fraction of test set for validation during training
	colorspace: str = "grayscale",	#grayscale|lum_lab: colorspace we want to work in
	kUseBigFile: bool = True
) -> tuple:
	#print("===(load_cifar)===")
	cifardir = projconfig.getCIFAR10Folder()

	if not kUseBigFile:
		#1: DL the original cifar10 and convert to 'colorspace:
		cifar10_train, cifar10_test = download_cifar(cifardir, colorspace = colorspace)
	else:
		#1: DL the original cifar10 as rgb:
		training_set, test_set = download_cifar(cifardir, colorspace = "rgb")
		testfile, trainfile = loadCIFAR.cifar10BigDb #datasets_root/"cifar-test.dat", datasets_root/"cifar-train.dat"
		
		#2: convert to binary chunk files:
		result = make_bigfiles(test_set, training_set, testfile, trainfile, colorspace = colorspace)
		assert(result)

		#3: open the chunk files and wrap it with CustomCIFARDB_lum to make it "lum_rgb" db
		cifar10_train, cifar10_test = colorSmartBigFiles(trainfile, testfile, colorspace, training_set, test_set)
		
		cifar10_train = CustomCIFARDB(cifar10_train, colorspace = colorspace)
		cifar10_test = CustomCIFARDB(cifar10_test, colorspace = colorspace)

		# print(f"{cifar10_train[0][0].shape=}")
		# print(f"{type(cifar10_train)=}")
	training_set, test_set = trainutils.getTrainTest(trset, cifar10_train, cifar10_test)	
	
	#1: subset our training set?
	#training_set = datasetutils.getBalancedSubset(training_set, 0.2)
	#1.1: subset our validate set?
	validateset = datasetutils.getBalancedSubset(test_set, validate)		#6k (assuming 60Kxx) validate set for frequent validate

	return training_set, test_set, validateset


if (__name__ == '__main__'):
	projconfig.createCIFAR10Folder()
	cifardir = projconfig.getCIFAR10Folder()
	print(cifardir)
	cifardir = projconfig.getCIFAR10Folder()
	print(f"{cifardir} {dirutils.direxist(cifardir)}")

	if True:
		trainingset = load_cifar("train", validate=0.2)
		training_set, test_set, validateset, trainTransform, testTransform = trainingset
		cifar10_train = training_set
		cifar10_test  = test_set
	else:
		cifar10_train = CustomCIFARDB(datasets.CIFAR10(cifardir, train = True, download = True, transform = PIL2graynumpy))
		cifar10_test = CustomCIFARDB(datasets.CIFAR10(cifardir, train=False, download = True, transform = PIL2graynumpy))

		ourTransform = dbaugmentations.ourAugmentations(*cifar10_train.getstats())
		print(ourTransform)

	print(f"train {len(cifar10_train)}, test {len(cifar10_test)}")
	print(cifar10_train[10])

	assert(issubclass(type(trainTransform), augmentation.Base))
	print(type(trainTransform))
