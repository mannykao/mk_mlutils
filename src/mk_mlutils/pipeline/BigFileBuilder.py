# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import argparse, os, pickle
import abc, copy
import time
import numpy as np
import gzip, struct
from io import BytesIO

#import our packages
from mkpyutils import dirutils, folderiter
from mk_mlutils.dataset import dataset_base

#from . import BigFile
from . import BigFile


def time_spent(tic1, tag='', count=1):
	toc1 = time.process_time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)/count:.2f}s")
	return

def writeXformedByEntry(
	bigfile: BigFile.BigChunk, 
	size_sum,	#track # of bytes written
	dataset, 
	xformed, 
	threshold=BigFile.noop_threshold, 
	kPickle=False,
	kZip=True
):
	num_entries = len(dataset)
	stream = BytesIO()
	offsets = []
	labels = []
	entries  = 0  	#track # of entries written

	#writing files. 
	for i, entry in enumerate(dataset):
		image, label = entry
		coeffs = xformed[i]
		coeffs_th = threshold(coeffs)	# thresholding the Shearlets coeffs.

		if (i % 10000) == 0:
			print(i)

		#writing data to temporary stream and getting the binary value.
		BigFile.reset(stream)

		if kZip:
			with gzip.GzipFile(fileobj = stream, mode = 'wb') as f:
				np.save(f, coeffs_th, allow_pickle = kPickle)		#**NOTE: do not allow pickle to avoid saving Python objects - only ndarray
		else:	
			np.save(stream, coeffs_th, allow_pickle = kPickle)		#**NOTE: do not allow pickle to avoid saving Python objects - only ndarray
		
		zipped = stream.getvalue() 	#this is the gzipped 'filename' in a string of bytes
		#writing the binary data to the merged file.
		sizeB = bigfile.write_data(bigfile.file, zipped)	## of bytes written
		size_sum += sizeB
		entries += 1

		offsets.append(sizeB)
		labels.append(label)
	assert(entries == num_entries)

	return offsets, labels	

def writeXformed1Chunk(
	bigfile: BigFile.Big1Chunk, 
	size_sum,	#track # of bytes written
	dataset, 
	xformed, 
	threshold=BigFile.noop_threshold, 
	kPickle=False,
	kZip=True
):
	num_entries = len(dataset)
	stream = BytesIO()
	offsets = []
	labels = dataset_base.getLabels(dataset)
	entries  = 0  	#track # of entries written

	BigFile.reset(stream)

	np.save(stream, xformed, allow_pickle = kPickle)		#**NOTE: do not allow pickle to avoid saving Python objects - only ndarray
		
	zipped = stream.getvalue()
	#writing the binary data to the merged file.
	pos = bigfile.file.tell()
	sizeB = bigfile.write_data(bigfile.file, zipped)	## of bytes written
	size_sum += sizeB
	entries += num_entries
	#print(f"bigfile.write_data pos={pos}, {sizeB} bytes")

	offsets.append(sizeB)

	return offsets, labels

kWriteXformed_dispatch = {
	"BigChunk":  writeXformedByEntry,
	"Big1Chunk": writeXformed1Chunk,
}

def writeBigFile(
	bigfile,		#Out: output BigFile object (should be empty but opened)
	bigfileclass:str, 	#= "BigChunk"
	dataset,		#original image dataset (CustomDataSet) 
	xformed, 		#source of xformed shearlets from BigFileBuilder.start()
	threshold=BigFile.noop_threshold,
	kPickle=False,
	kZip=True
):
	""" the main loop """
	assert(bigfile.isopen)
	num_entries = len(dataset)
	#print(f"  len(dataset) {num_entries}")
	entries  = 0  	#track # of entries written
	size_sum = 0	#track # of bytes written
	stream = BytesIO()
	offsets = []
	labels = []
	fout = bigfile.file

	#writing initial header.
	size_sum += bigfile.write_header(fout, num_entries, 0, 0, kZip, 0)

	writeXform = kWriteXformed_dispatch.get(bigfileclass, None)
	assert(writeXform is not None)
	offsets, labels = writeXform(bigfile, size_sum, dataset, xformed, threshold, kPickle, kZip)

	#offsets.append(size_sum)
	#print(f'address after file {i} : ', fout.tell())
		
	return  offsets, labels

def ourIdentityXform(dataset, n_jobs):
	""" a no-op xform - iterate 'dataset' to convert it to an ndarray of images """
	coeffs = [entry.coeffs for entry in dataset]
	return np.asarray(coeffs)

class BigFileBuilder():
	""" Builder to apply 'xform' to each element of 'dataset' passed to .start and
		gzip each to form an indexable file. __getitem__ will perform the unzipping.
	"""
	kBigFileClasses = {
		"BigChunk":  BigFile.BigChunk,
		"Big1Chunk": BigFile.Big1Chunk,
	}

	def __init__(self, 
		filename: str,
		bigfileclass:str = "BigChunk",
		xform=ourIdentityXform, 
		n_jobs=4,
	):
		self._filename = filename
		self.bigfileclass = bigfileclass
		self.xform = xform
		self.n_jobs = n_jobs

	@property
	def filename(self):
		return self._filename

	@property
	def targetclass(self):
		return BigFileBuilder.kBigFileClasses.get(self.bigfileclass, None)

	def start(self, dataset):
		print(f"mk_mlutils building BigFile for {self._filename}..", flush=True)
		self.num_entries = len(dataset)
		self.now = time.time()
		targetclass = self.targetclass
		assert(targetclass is not None)
		self.bigfile = targetclass(self.filename, 'wb', kOpen=True)
		assert(self.bigfile.isopen)	#TODO: try|except

		#the parallel .apply where the xform is applied to the input dataset
		xformed = self.xform(dataset, self.n_jobs)

		return self.bigfile, xformed

	def finalize(self, bigfile, offsets, labels, opt_threshold, zipped, colorcode):
		"""	patch header with the offset array addresses """
		finalizeEx(bigfile, self.num_entries, offsets, labels, zipped, colorcode)

		then = time.time()
		print(f' time: {(then - self.now):.2f} secs')

	def doit(self, 
		dataset,
		threshold=BigFile.noop_threshold,
		verify=True,
		kPickle=False,
		kZip=True,
		colorspace=None,
	):
		bigfile, xformed = self.start(dataset)

		offsets, labels = writeBigFile(
			bigfile, 			#Out: output file handle
			self.bigfileclass,
			dataset,		#original image dataset (CustomDataSet) 
			xformed, 		#source of xformed shearlets
			threshold,
			kPickle=kPickle,
			kZip=kZip
		)
		colorspace = colorspace if colorspace else dataset.colorspace
		colorcode = BigFile.colorspace2code(colorspace)
		self.finalize(bigfile, offsets, labels, threshold, kZip, colorcode)

		#TODO: use mpverify instead of serial code
		if verify:
			self.bigfile.openfile(mode='rb')
			BigFile.verifyBigFile(self.bigfile, dataset, xformed, threshold)

def combine_offsets(offsets):
	""" offsets[2] is a list of tuples where each tuple is an offset table from BigFile """	
	offsets0 = offsets[0]
	offsets1 = offsets[1]
	end0 = offsets0[-1]		#offset -> end of table 0
	end1 = offsets1[-1]		#offset -> end of table 1
	print(f"{end0}, {end1}")

	#1: first method - convert to lists
	#result = list(offsets0[0:-1])
	#adjusted = [entry + end0 for entry in (offsets1[0:-1])]
	#result.extend(adjusted)

	#2: use tuples directly
	#result = offsets0[0:-1]
	#adjusted = (entry + end0 for entry in offsets1)
	#result += tuple(adjusted)
	#print(f"len(result) {len(result)}")

	sizes0 = BigFile.BigChunk.offsets2sizes(offsets0)
	sizes1 = BigFile.BigChunk.offsets2sizes(offsets1)
	newsizes = sizes0 + sizes1

	reference = sizes0 + sizes1
	for i, size in enumerate(newsizes):
		ref = reference[i]
		if ref != size:
			print(f"failed[{i}] {size}, {ref}")
			assert(False)

	#result = BigFile.DiskShearletDataset.sizes2offsets(newsizes)
	return newsizes		#write_offsets expect sizes

def finalizeEx(bigfile, num_entries, offsets, labels, zipped, colorcode):
	"""	patch header with the offset array addresses """
	fout = bigfile.file
	#updating header, offsets and labels.
	offset_add = bigfile.write_offsets(fout, offsets)
	labels_add = bigfile.write_labels(fout, labels)

	bigfile.write_header(fout, num_entries, offset_add, labels_add, zipped, colorcode)
	bigfile.finalize()
	
	#print("Final Header:", num_entries, offset_add, labels_add)
