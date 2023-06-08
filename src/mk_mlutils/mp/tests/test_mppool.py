# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
import os, sys, time
import multiprocessing
import numpy as np
#import asyncio 
from queue import Empty, Full
#import signal

#our packages
from datasets import dataset_base as db

from mk_mlutils import projconfig, torchutils
#from mk_mlutils.dataset import dataset_base as db
from mk_mlutils.dataset import fashion
from mk_mlutils.pipeline import BigFile
from mk_mlutils.mp import mppool


kBigFile=True 	#use BigFile as input dataset

"""
	Unit test for BigFile (BigChunk & DiskShearletDataset) being marshallable across processes.
"""

class Worker(multiprocessing.Process):
	instanceId = 0

	def __init__(self, queue, send_end, workerargs):
		""" This will be executed in the parent process """
		print(f"Worker() pid {os.getpid()}", flush=True)	#Note: this will be parent's pid		
	
		super().__init__()
		self.queue = queue
		self.send_end = send_end
		self.workerargs = workerargs
		self.instanceId = Worker.instanceId
		Worker.instanceId += 1

	def onceInit(self, kCUDA=False):
		""" per-worker|per-core once Init - we are in the worker process """
		print(f" Worker.onceInit: pid {os.getpid()}", flush=True, end='')

		kCUDA = self.workerargs.get('CUDA', False)
		self.device = torchutils.onceInit(kCUDA=kCUDA)
		self.dataset = prepWorker(self.workerargs)

	@classmethod
	def qNotDone(cls, work):	
		return work is not mppool.MPPool.kSTOP_VALUE

	def oneChunk(self, sh_xform, work):
		""" process 1 chunk of work """
		#print(f" **oneChunk {work}")	
		start, end = work
		dbchunk = db.CustomDatasetSlice(self.dataset, ourslice=(start, end-start))
		shearlets = [] 	#xformdataset_serial(sh_xform, dbchunk, work)

		return (work, shearlets)

	@staticmethod
	def checkChunkOutput(output):
		result = (type(output) is tuple)		
		result &= len(output) == 2
		assert(result)
		return result

	def run(self):
		""" the multiprocessing.Process.run()
			We pull items off the shared queue until we see 'kSTOP_VALUE'
		"""
		q = self.queue
		workerargs = self.workerargs
		shfactory = workerargs['shfactory'] 
		kCUDA = workerargs['CUDA']

		#1: per-core/worker once init:
		self.onceInit(kCUDA=kCUDA)

		sh_xform = None

		results = []
		#2: persistent thread
		while True:
			work = q.get()

			if Worker.qNotDone(work):		#are we all done?
				print(f" work {work}") 

				#3: record the results in a tuple to be returned to caller
				output = self.oneChunk(sh_xform, work)
				#self.checkChunkOutput(output)
				results.append(output)					
			else:
				result_tuple = (self.instanceId, results)
				self.send_end.send(result_tuple)
				results = []

				#time.sleep(5)

				print(f"  poison-pill[{self.instanceId}]")
				break	#poison-pill
#end class Worker

def time_spent(tic1, tag='', count=1):
	toc1 = time.time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/count:3f}s")
	return

def prepWorker(workerargs):
	tic1 = time.time()
	full_dataset = workerargs['full_dataset']
	print(f" prepWorker {full_dataset}")
	full_dataset.openfile()
	assert(len(full_dataset))
	time_spent(tic1, f"prepWorker ", count=1)

	return full_dataset

class XformPool(mppool.MPPool):	
	def makeWorker(self, 
		queue:multiprocessing.Queue, 
		send_end:multiprocessing.Pipe, 
		workerargs:dict
	) -> multiprocessing.Process:
		""" create one Worker """
		return Worker(queue, send_end, workerargs)	#this will start running immediately


if (__name__ == '__main__'):
	from mk_mlutils import coshrem_xform

	#1. data paths definition.
	#datafolder = "../covid-chestxray-dataset/"
	#data_images = datafolder + "images"
	#meta_data = datafolder + "metadata.csv"

	cifarfolder = projconfig.getCIFAR10Folder()
	cifar_big = cifarfolder / 'cifar-testX.dat'

	#2. load our gzipped chunk dataset
	full_dataset = BigFile.Big1Chunk(cifar_big, kOpen=False, colorspace = "lab")

	start = 0
	end   = 16  #len(full_dataset)
	#dbchunk = db.CustomDatasetSlice(full_dataset, ourslice=(start, end-start))

	#3: worker's parameter block - must be marshallable (pickle)
	shfactory = (coshrem_xform.ksh_spec, coshrem_xform.CoShXform)

	workerargs = {
	#	'data_images' : data_images,
	#	'meta_data'	 : meta_data,
	#	'bigzipped'	 : full_dataset,
		'full_dataset' : full_dataset,
		'chunkfactor': 3,		#divide work finer than n_jobs for better load-balance
								#1: 23.5, 2: 24.6, 3: 22.8, 4: 22.5 		
		'shfactory'	 : shfactory,
		'CUDA'		 : False	#gpu: 22s,   cpu: 43.2s (xvr)
								#gpu: 15.8s, cpu: 35.2s (bigzipped)
	}
	#4.1: create mp pool
	mypool = XformPool(4)

	tic1 = time.time()
	#asyncio.run( mypool.doit(workerargs, len(dbchunk), kPersist=False) )
	mypool.doit(workerargs, end-start, kPersist=False)

	time_spent(tic1, f"XformPool ", count=1)

	results = mypool.results
	#print(len(results), type(results[0]), f"results[0].shape {results[0].shape}")
	print(len(results), type(results), results)

	print("This is a noop template for MPPool to demonstrate how to setup the client code")

"""
Worker() pid 11540
Worker() pid 11540
Worker() pid 11540
Worker() pid 11540
 pid[11540] using 4 cores to process 16
chunk=(0, 2), chunk=(2, 4), chunk=(4, 6), chunk=(6, 8), chunk=(8, 10), chunk=(10, 12), chunk=(12, 14), chunk=(14, 16), 
 Worker.onceInit: pid 5896 Worker.onceInit: pid 10644 Worker.onceInit: pid 25260 Worker.onceInit: pid 12672torchutils.onceInit device = cpu
initSeeds(1)
 prepWorker Big1Chunk(D:\Dev\SigProc\onsen\data\CIFAR10\cifar-testX.dat)
time spend on prepWorker  method = 0.140382s
 work (0, 2)
 work (2, 4)
 work (4, 6)
 work (6, 8)
 work (8, 10)
 work (10, 12)
 work (12, 14)
 work (14, 16)
  poison-pill[3]
torchutils.onceInit device = cpu
initSeeds(1)
 prepWorker Big1Chunk(D:\Dev\SigProc\onsen\data\CIFAR10\cifar-testX.dat)
time spend on prepWorker  method = 0.151334s
  poison-pill[2]
torchutils.onceInit device = cpu
initSeeds(1)
 prepWorker Big1Chunk(D:\Dev\SigProc\onsen\data\CIFAR10\cifar-testX.dat)
time spend on prepWorker  method = 0.173236s
  poison-pill[0]
torchutils.onceInit device = cpu
initSeeds(1)
 prepWorker Big1Chunk(D:\Dev\SigProc\onsen\data\CIFAR10\cifar-testX.dat)
time spend on prepWorker  method = 0.179214s
  poison-pill[1]
time spend on XformPool  method = 2.364834s
8 <class 'list'> [((0, 2), []), ((2, 4), []), ((4, 6), []), ((6, 8), []), ((8, 10), []), ((10, 12), []), ((12, 14), []), ((14, 16), [])]
This is a noop template for MPPool to demonstrate how to setup the client code
MPPool.__del__
[Finished in 4.5s]
"""
	