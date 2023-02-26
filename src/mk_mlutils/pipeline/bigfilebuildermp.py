# -*- coding: utf-8 -*-

import functools, pickle
import numpy as np
import os, sys, time
from collections import Counter

#our packages
from . import BigFile, BigFileBuilder, combine_sh
from .. import torchutils 
#our modules
#import loadFashion, shsys_def
#import 

class BigFileBuilderMP():
	""" parallel applying 'xform' to chunks of 'dataset' -> BigFile in each worker """
	def __init__(self,
		filename, 
		xform,			#a factory for a subclass of ShXform (shnetutil.shxform )
		xform_config,	#a dict controlling the configuration for 'xform'
		chunksize=-1	#-1 means we will use the n_jobs to control chunksize
	):
		self.filename = filename
		self.xform = xform
		self.xform_config = xform_config
		self.n_jobs = xform_config.get('n_jobs', 4)
		self.target_chunksize = chunksize

	def start(self, dataset):
		""" usually this is where once only init is placed. For this class it is actually called
		at the start of 'doit' because we want it to be per-worker and not per-Builder
		"""
		print(f"BigFileBuilderMP: {self.filename}..")
		self.dataset = dataset
		self.num_entries = len(dataset)
		self.now = time.time()

		if (self.target_chunksize == -1):
			kNumJobs = self.n_jobs
			self.target_chunksize = int((self.num_entries + kNumJobs-1) // kNumJobs)
			print(f"target_chunksize {self.target_chunksize}")
		return 

	def doit(self, dataset, threshold, verify=False):
		#the parallel self.xform where the xform is applied to the input dataset

		#1: once init
		self.start(dataset)

		#2: apply xform in parallel to the whole dataset (any chunking happens in prep or in dispacher)
		shfiles = self.xform(**self.xform_config)	#typically this is mpxform.xformdataset
		#print(f" doit:self.xform finished..")

		self.finalize(shfiles, dataset, threshold, verify)

	def finalize(self, shfiles, dataset, threshold, verify=False):
		""" combine all the worker's work into a merged BigFile """

		#1: combine the <n>.npy files into the BigFile:
		#Note: this is the serial and the slowest step. Is being replaced by combine_big.sh
		bigfile, shearlets = combine_sh.combine2BigFile(self.filename, dataset, shfiles, threshold)
		bigfile.finalize()

		if verify:	#deprecated - use mpverify for this
			print("BigFile.verifyBigFile..")
			BigFile.verifyBigFile(
				self.filename,
				self.dataset, 
				shearlets, 
				threshold=threshold
			)

def checkChunks(chunks):
	sizes = []
	for chunk in chunks:
		start, end, shfilename, *_ = chunk	
		sizes.append(end-start)
	cnts = Counter(sizes[0:-1])

	return (len(cnts) == 1), list(cnts.keys())[0]		#same size for all chunks	

def fromChunks(chunks, numentries):
	numentries = numentries
	_, chunksize = checkChunks(chunks)

	ourchunks = []
	chunkoffsets = []
	for chunk in chunks:
		start, end, shfilename, *_ = chunk	
		ourchunks.append(BigFile.BigChunk(shfilename))
		chunkoffsets.append(start)
	assert(numentries == end)
	return ourchunks, chunksize	

def verifyBigChunks(ourchunks, bigfile_chunks):
	print("verifyBigChunks")
	loc = 0		
	for chunk in ourchunks:
		print(f" chunk {chunk.filename}..")
		for i in range(len(chunk)):
			img0, label0 = chunk[i]
			img1, label1 = bigfile_chunks[loc]
			assert(label0 == label1)				
			loc += 1	

def removePath(chunks):
	output = []
	for chunk in chunks:
		start, end, filepath = chunk
		filename = os.path.basename(filepath)
		output.append((start, end, filename))
	print(f"removePath {output}")	
	return output

class BigFileBuilder2Chunks(BigFileBuilderMP):
	""" build a composite BigFile from a list of chunks (each is an old BigFile) """
	def __init__(self,
		filename, 
		xform,			#a factory for a subclass of ShXform (shnetutil.shxform )
		xform_config,	#a dict controlling the configuration for 'xform'
		chunksize=-1	#-1 means we will use the n_jobs to control chunksize
	):
		print(f"BigFileBuilder2Chunks: {filename}.. ", end='')
		super().__init__(filename, xform, xform_config)

	def start(self, dataset):
		super().start(dataset)

	def finalize(self, shfiles, dataset, threshold, verify=False):
		print(f" BigFileBuilder2Chunks.finalize({shfiles}, {len(dataset)})")
		bigfile_chunks = BigFile.DiskShearletDataset(self.filename, "wb")

		result, chunksize = checkChunks(shfiles)
		assert(result)
		print(f" chunksize {chunksize}")

		#bigfile_chunks.fromChunks(shfiles, len(dataset))
		ourchunks, chunksize = fromChunks(shfiles, len(dataset))

		#hack into bigfile for testing
		bigfile_chunks.chunksize = chunksize
		bigfile_chunks.chunks = ourchunks

		shfiles = removePath(shfiles)
		bigfile_chunks.finalize(shfiles)

		if verify:
			verifyBigChunks(ourchunks, bigfile_chunks)

		print(f" bigfile_chunks.size({len(bigfile_chunks)})")
		return bigfile_chunks
