# -*- coding: utf-8 -*-
"""

Title: Use MP to perform our data xform/preconditioning.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""
import argparse, pickle
import numpy as np
import os, sys, time

from . import BigFile, BigFileBuilder

# https://www.peterbe.com/plog/concurrent-gzip-in-python

def identity(dataset, n_jobs, shearlets):
	""" a NULL xform - just return the shearlets passed to it """
	return shearlets

def	cleanup(shfiles):
	for num, file, *_ in shfiles:
		os.remove(file)

def combine2BigFile(mergedfile, dataset, shfiles, threshold=BigFile.opt_threshold):
	""" combine the <n>.npy into 1 BigFile 'mergedfile' """
	#1: the len of shfiles is equal to kNumJobs in mpxform, each entry is a 4tuple
	shfiles = sorted(shfiles, key=lambda file: file[0])
	print(shfiles)

	shearlets = []
	for num, file, maxerr, minerr in shfiles:
		print(f" shfilename '{file}', minerr={minerr:6f}, maxerr={maxerr:6f}", end='')
		with open(file, "rb") as f:
			coeffs = np.load(f, allow_pickle=False)	
			shearlets.append(coeffs)
			print(coeffs.shape)
	total = np.concatenate(shearlets, axis=0)
	print(total.shape)		

	builder = BigFileBuilder.BigFileBuilder(
		mergedfile, 
		#use closure to pass 'total'
		lambda dataset, n_jobs: identity(dataset, n_jobs, total),
		n_jobs=4
	)
	bigfile = builder.doit(dataset, threshold=threshold)

	cleanup(shfiles)

	return bigfile, total

if __name__ == '__main__':
	import loadFashion

	parser = argparse.ArgumentParser(description='BigFile.py')
	parser.add_argument('-dataset', type=str, default='test', help='test or train')
	args = parser.parse_args()

	kUseTrain = (args.dataset == 'train')

	root = '../data/'
	outfolder = '../output/'
	shfilenames = outfolder + 'shfiles.dat'
	mergedfile = outfolder + 'test-set-tmp.dat'

	device, dataset = loadFashion.onceInit(root, istrain=kUseTrain, kCUDA=False)
	print(dataset, type(dataset), {type(dataset[0])})

	with open(shfilenames, "rb") as f:
		shfiles = pickle.load(f)

	#change "ooutput/0.npy" -> "../output/0.npy"	
	shfiles = [(i, j, k, l) for i, j, k, l in shfiles]

	bigfile, shearlets = combine2BigFile(mergedfile, dataset, shfiles)
