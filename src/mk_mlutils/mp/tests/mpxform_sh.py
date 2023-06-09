# -*- coding: utf-8 -*-
"""

Title: Use MP to perform our data xform/preconditioning.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""
import os, sys, time
import functools, pickle
import logging
import multiprocessing
from queue import Empty, Full
import numpy as np
import signal

#our packages
from mkpyutils import dirutils
from mkpyutils import testutil

from datasets import dataset_base
import datasets.fashionmnist as fashion

from mk_mlutils.pipeline import batch, augmentation
from mk_mlutils import coshrem_xform, torchutils
from mk_mlutils.mp import mpxform, mppool

#last update: 11//22/2021. Ran successfully

kSerial=True

def time_spent(tic1, tag='', count=1):
	""" measure CPU process time - not wall time """
	toc1 = time.process_time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/(count*10):.2f}s")
	return

def getBatch(dataset, mybatch):
	return [dataset[b].coeffs for b in mybatch]

def xformdataset_serial(
	shfactory:tuple,
	device,	
	dataset,
	work		#(start, end)
) -> list:
	""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
	print(f"xformdataset_serial {work}..")
	batchsize = 512

	sh_spec, xform_factory = shfactory
	#print(f"worker {sh_spec=}, {xform_factory=}")
	assert(issubclass(xform_factory, augmentation.CoShREM))
	sh_xform = xform_factory(tocplx = True)

	auglist = [
		augmentation.Pad([(0,0), (2,2), (2,2)]),		  
		sh_xform,
#		augmentation.CoShREM(tocplx = True, denoise = False),
	]
	ourTransform = augmentation.Sequential(auglist)

	shearlets = []
	count = 1
	tic1 = time.time()
	print(f" {sh_xform} using {1} cores")
	
	batchbuilder = batch.BatchBuilder(dataset=dataset, batchsize=batchsize, shuffle=False)
	epoch = batchbuilder.epoch()
	#2.1: iterate all the batches and apply our xform to enable them to be captured in 'capturecache'
	for b, mybatch in enumerate(epoch):
		#imglist, labels = batch.getBatchAsync(dataset, mybatch) 	#we are on 1 core, do not use asyncio
		imglist = getBatch(dataset, mybatch)
		imglist = ourTransform(imglist)
		shearlets.append(imglist)
		#labels.append(label)

	testutil.time_spent(tic1, "", count=1)
	print(type(shearlets), len(shearlets))

	return shearlets

def worker(
	#start of fixed part of the signature
	jobid,
	queue, 
	send_end, 
	shfactory,	#(sh_spec, xform_obj(subclass of ShXform()))
	#end of fixed part of the signature, start of the variable part..
	workerargs
):
	outfolder = workerargs['outfolder'] 
	initfunc  = workerargs['onceInit']
	sh_spec, xform_factory = shfactory

	#0: check a few parms
	assert((type(outfolder) is str))	#and (type(datasets) is str)
	assert(issubclass(xform_factory, augmentation.CoShREM))
	#1: per-worker init
	n_jobs = 1
	device, dataset = initfunc(jobid, shfactory, workerargs)

	try:
		work = queue.get(block = True)	#TODO: time out
		print(f"{jobid}: {work}")
		
		start, end = work
		dbchunk = dataset_base.CustomDatasetSlice(dataset, ourslice=(start, end-start))
		bigfilename = outfolder + f"{jobid}.dat"

		xformdataset_serial(shfactory, device, dbchunk, work)

		send_end.send((start, end, bigfilename))

	except Empty:
		pass
	return

def prepFashion(	
	jobid,
	shfactory,	#(sh_spec, xform_obj(subclass of ShXform()))
	workerargs,
) -> tuple:
	""" the once init functor for each worker in mpxform for Fashion """
	trset = workerargs['trset'] 
	#1: init pytorch to run on the device of choice
#	device = torchutils.onceInit(kCUDA=True)
	device = 'cpu'

	#2: load our Fashion dataset
	train, test, validate = fashion.load_fashion(trset=trset, kTensor=False, validate=0.2)
	dataset = train
	#print(type(dataset), {type(dataset[0])}, {len(dataset)})

	return device, dataset

def testSerial(shfactory:tuple, workerargs):
	tic1 = time.time()

	device, dataset = prepFashion(0, shfactory, workerargs)
	start, end = 0, len(dataset)
	dbchunk = dataset_base.CustomDatasetSlice(dataset, ourslice=(start, end-start))
	work = (start, end)

	xformdataset_serial(shfactory, device, dbchunk, work)

	testutil.time_spent(tic1, "testSerial", count=1)

def main():
	outfolder = 'output/'
	kUseTrain = False
	trset = 'train'		#load the 10k test
	train, test, validate = fashion.load_fashion(trset=trset, kTensor=False, validate=0.2)
	#print(fashion)

	#1: request the test set (10k)
	dataset = train
	print(f"dataset {type(dataset)=}, {len(dataset)=}")

	cosh_xform = coshrem_xform.CoShXform

	#1: package up our xform factory
	shfactory = (coshrem_xform.ksh_spec, augmentation.CoShREM)
	workerargs = {
		'trset'	 	: trset,
		'device'	: 'gpu',
		'onceInit'	: prepFashion,		#per-worker once only init
		'outfolder' : outfolder,
	}
	if kSerial: 	#27.93s
		testSerial(shfactory, workerargs)
		print("")
		
	#mpform: 15.80s
	tic1 = time.time()

	shfiles = mpxform.xformdataset(
		dataset,		#a CustomDataset
		shfactory,		#(sh_spec, pysh_xform.PyShXform) 
		None,			#outfolder+'shfiles.dat'
		worker,			#worker function
		workerargs,		#args for the worker
		n_jobs=4,
	)
	testutil.time_spent(tic1, "mpxform.xformdataset", count=1)


if __name__ == '__main__':
	main()


""" 11/22/2021
dataset type(dataset)=<class 'shnetutil.pipeline.loadMNIST.FashionDataset'>, len(dataset)=60000
mpxform.xformdataset 'None', {'rows': 256, 'cols': 256, 'scales_per_octave': 2, 'shear_level': 3, 'octaves': 1, 'alpha': 0.8}
D:\Dev\SigProc\onsen\data\FashionMNIST\raw
0: (0, 15000)
xformdataset_serial (0, 15000)..
 CoShXform({'rows': 32, 'cols': 32, 'scales_per_octave': 2, 'shear_level': 3, 'octaves': 1, 'alpha': 0.5}) using 1 cores
time spend on  method = 12.47s
<class 'list'> 30
D:\Dev\SigProc\onsen\data\FashionMNIST\raw
1: (15000, 30000)
xformdataset_serial (15000, 30000)..
 CoShXform({'rows': 32, 'cols': 32, 'scales_per_octave': 2, 'shear_level': 3, 'octaves': 1, 'alpha': 0.5}) using 1 cores
time spend on  method = 12.47s
<class 'list'> 30
D:\Dev\SigProc\onsen\data\FashionMNIST\raw
3: (30000, 45000)
xformdataset_serial (30000, 45000)..
 CoShXform({'rows': 32, 'cols': 32, 'scales_per_octave': 2, 'shear_level': 3, 'octaves': 1, 'alpha': 0.5}) using 1 cores
time spend on  method = 12.45s
<class 'list'> 30
D:\Dev\SigProc\onsen\data\FashionMNIST\raw
2: (45000, 60000)
xformdataset_serial (45000, 60000)..
 CoShXform({'rows': 32, 'cols': 32, 'scales_per_octave': 2, 'shear_level': 3, 'octaves': 1, 'alpha': 0.5}) using 1 cores
time spend on  method = 12.50s
<class 'list'> 30
shfiles(piped): [(0, 15000, 'output/0.dat'), (15000, 30000, 'output/1.dat'), (30000, 45000, 'output/3.dat'), (45000, 60000, 'output/2.dat')]
Elapsed time: xformdataset() 15.797355s
time spend on mpxform.xformdataset method = 15.80s
[Finished in 18.5s]
"""
