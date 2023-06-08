# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal 

"""
#import logging
import abc
from typing import List, Tuple, Union, Optional
import multiprocessing
#import asyncio 
from queue import Empty, Full
#import signal
import os, sys, time

#our packages
#our modules

kUnitTest=True

#
# Python Pool.map() in theory solves the same problem as MPPool(). 
# The reason we did not adopt:
#   1. no concept of per-core/worker onceInit()
#	2. work has to be marshalled across processes:
#	   work = (["A", 5], ["B", 2], ["C", 1], ["D", 3]) 		#["A", 5] is one slice of our giant dataset
#	   p = Pool(2)
#	   p.map(worker, work)		#'worker' and 'work' are marshalled 
#
# https://pymotw.com/2/multiprocessing/communication.html

def get_chunk(numentries:int, numjobs:int):
	chunk = int((numentries + numjobs-1)/numjobs)
	return chunk

class MPPool(metaclass=abc.ABCMeta):
	""" A pool of processes to run async workers with load balancing 
		- it behaves like persistent-threads where 
		  a) each core is occupied by one 'worker' created by makeWorker()
		  b) each worker pull work from a single shared queue and are otherwise blocked.
		  c) each worker sends its result using a Pipe (lighter than a Queue)
		  d) doit() method perform load-balancing by dividing the work into work-items, using
		     the concept of 'chunkfactor' - which is the the factor we mulitply n_jobs with to
		     divide the input into chunks. The chunks are pushed onto the queue.
		  e) each worker is invoked with the queue and its allocation of workitems.
		     Since these are marshalled across processes they should be compact and 
		     marshallable.
	Usage:	     	
		1) Derive from MPPool and implement the abstract method makeWorker() - e.g. XformPool.
		2) mp/tests/test_mppool.py 
	"""
	kSTOP_VALUE = None 		#our poison-pill to stop workers when popped from queue

	def __init__(self, poolsize=4):
		self.poolsize = poolsize	#number of process in our tool
		self.pool = []
		self.jobs = []
		self.queue = None
		self.pipe_list = None
		self.send_list = None

	def __del__(self):
		print("MPPool.__del__")
		#if finalize() was called with kJoin all the processes should have terminated
		self.stopWorkers()

		for j in self.jobs:
			if j.is_alive():
				print(f"join {j}")	
				j.join()		

	@property
	def qInit(self):
		""" predicate for onceInit has not been called """
		return self.pipe_list is None

	@abc.abstractmethod
	def makeWorker(self, 
		queue:multiprocessing.Queue, 
		send_end:multiprocessing.Pipe, 
		workerargs:dict
	) -> multiprocessing.Process:
		""" implemened by client code """
		pass		

	def onceInit(self, workerargs:dict):
		""" per-pool once only init """
		assert(type(workerargs) == dict)
		queue = multiprocessing.Queue()
		self.queue = queue

		if self.qInit:
			jobs = []
			pipe_list = []
			send_list = []
			for i in range(self.poolsize):
				recv_end, send_end = multiprocessing.Pipe(False)
				pipe_list.append(recv_end)
				send_list.append(send_end)
				#TODO: see if streams are better https://docs.python.org/3/library/asyncio-stream.html#register-an-open-socket-to-wait-for-data-using-streams
			
				p = self.makeWorker(queue, send_end, workerargs)
				assert(isinstance(p, multiprocessing.Process))

				jobs.append(p)
				p.start()		#start the worker - mostly will block waiting for work

			self.jobs = jobs
			self.pipe_list = pipe_list
			self.send_list = send_list

	def stopWorkers(self):
		for p in self.jobs:
			if p.is_alive():
				self.queue.put(MPPool.kSTOP_VALUE)		#poison-pill to tell workers to break out

	def scheduleWork(self, numentries:int, workerargs:dict, kPersist=False):
		""" divide the input into 'chunkfactor' sized workitems and push onto the queue """
		print(f" pid[{os.getpid()}] using {self.poolsize} cores to process {numentries}", flush=True)

		queue = self.queue
		self.numentries = numentries

		chunkfactor = workerargs.get('chunkfactor', 3)
		chunk = get_chunk(numentries, self.poolsize*chunkfactor)

		for c in range(0, numentries, chunk):
			start = c
			end = min(numentries, c+chunk)
			queue.put((start, end))
			print(f"chunk={(start, end)}, ", end='')
		print()

		#3: 'poison-pill' => no more work tell workers to quit
		if not kPersist:
			self.stopWorkers()

	def doit(self, workerargs:dict, numentries:int, kPersist:bool=False):
		""" initiated the parallel processing and wait for all workers """
		self.onceInit(workerargs)
		self.scheduleWork(numentries, workerargs, kPersist)
		
		#4: Wait for the worker to finish and collect results from each worker
		self.results = self.finalize(kPersist)

	def finalize(self, kPersist:bool):
		""" Wait for the worker to finish and collect results from each worker """
		results = [x.recv() for x in self.pipe_list]

		self.queue.close()
		self.queue.join_thread()

		self.verify(results)

		#sort results to reconstruct original order
		tmplist = []
		for result in results:
			instid, output = result

			#2: loop through per-worker output list
			tmplist.extend(output)
		results = sorted(tmplist, key=lambda e: e[0][0])	

		if not kPersist:
			#TODO: try not to close the queue for persistent case, just need to test
			for p in self.jobs:
				p.join()
	
		return results

	def verify(self, results):
		""" placeholder for client code to do final verification - called from finalize() """
		pass

if kUnitTest:
	import mk_mlutils.dataset.fashion as fashion
	import mk_mlutils.utils.torchutils as torchutils
	from datasets import dataset_base

	def xformdataset_serial(
		xform,
		dataset:dataset_base.DataSet,
		work:tuple		#(start, end)
	) -> list:
		""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
		print(f"xformdataset_serial {work}..")
		assert(type(dataset))

		#shearlets = []
		labels = []
		
		start, end = work
		for i in range(0, end-start):
			item = dataset[i]
			img, label = item

			#coeffs = sh_xform.xform(item)
			#shearlets.append(coeffs.cpu().numpy())
			labels.append(label)

		return labels

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
			print(f" self.dataset {self.dataset}")

		@classmethod
		def qNotDone(cls, work):	
			return work is not MPPool.kSTOP_VALUE

		def oneChunk(self, xform, work:tuple) -> tuple:
			""" process 1 chunk of work """	
			start, end = work
			dbchunk = dataset_base.CustomDatasetSlice(self.dataset, ourslice=(start, end-start))
			#shearlets = xformdataset_serial(xform, dbchunk, work)
		
			#do some processing on 'dbchunk'
			xformed = xformdataset_serial(xform, dbchunk, work)

			return (work, xformed)

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
	#		shfactory = workerargs['shfactory'] 
			kCUDA = workerargs['CUDA']

			#1: per-core/worker once init:
			self.onceInit(kCUDA=kCUDA)

	#		sh_spec, xform_factory = shfactory
	#		sh_xform = xform_factory(sh_spec)
	#		sh_xform.start(self.device)
	#		sh_sys = sh_xform.shearletSystem
			#shearlet_spec = sh_xform.sh_spec

			results = []
			#2: persistent thread
			while True:
				work = q.get()

				if Worker.qNotDone(work):		#are we all done?
					print(f" work {work}") 

					#3: record the results in a tuple to be returned to caller
					output = self.oneChunk(None, work)
					self.checkChunkOutput(output)
					results.append(output)					
				else:
					result_tuple = (self.instanceId, results)
					self.send_end.send(result_tuple)
					results = []
					print("  poison-pill")
					break	#poison-pill
#end class Worker

	class XformPool(MPPool):	
		def makeWorker(self, 
			queue:multiprocessing.Queue, 
			send_end:multiprocessing.Pipe, 
			workerargs
		):
			""" create one Worker """
			return Worker(queue, send_end, workerargs)	#this will start running immediately

		def finalize(self, kPersist:bool):
			results = super().finalize(kPersist)
			#remove the work tuples to make a final list of outputs 
			finallist = []
			[finallist.extend(lst[1]) for lst in results]
			return finallist

		def verify(self, results):
			total = 0
			#1: loop through workers
			for result in results:
				instid, output = result

				#2: loop through per-worker output list
				count = 0
				for entry in output:
					work, shearlets = entry
					count += work[1] - work[0]
					assert((work[1] - work[0]) == len(shearlets))
				total += count
			assert(total == self.numentries)
			return total == self.numentries

	def prepWorker(workerargs:dict):
		full_dataset = workerargs['full_dataset']
		full_dataset, test, validateset, *_ = fashion.load_fashion(full_dataset, validate=.3)
		return full_dataset			
#end... kUnitTest


if (__name__ == '__main__') and kUnitTest:
	train, test, validateset, *_ = fashion.load_fashion('train', validate=.3)
	print(f"{len(train)=}, {len(test)=}, {len(validateset)=}")

	workerargs = {
		'chunkfactor': 	3,		#scheduleWork() use this to divide work finer than n_jobs 
		'full_dataset': 'train',
								#for better load-balance
		'CUDA'		 : 	False,
	}
	pool = XformPool(poolsize=2)
	pool.doit(workerargs, len(train), kPersist=False)

	xformed_results = pool.results
	print(f"{len(xformed_results)=}")

	for i, item in enumerate(train):
		assert(xformed_results[i] == item.label)

	#explicit invoke the destructor to free resources	
	del pool
