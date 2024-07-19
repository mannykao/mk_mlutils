# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import argparse, os, sys, pickle
import time
import gzip, struct
from collections import namedtuple
import numpy as np
#import numba
from io import BytesIO
from collections import Counter

#import our packages
from mkpyutils import dirutils, folderiter, testutil
# our modules
from mldatasets import dataset_base


def time_spent(tic1, tag='', count=1):
	toc1 = time.process_time() 
	print(f"time spend on {tag} method = {(toc1 - tic1)*100./count:.2f}ms")
	return

def noop_threshold(coeff, lower = None, upper = None):
	return coeff

#@numba.jit(nopython=True)
def opt_threshold(coeff, lower = -0.0045709, upper = 0.0045709):
	ind_lower = coeff >= lower
	ind_upper = coeff <= upper
	indexes = np.logical_and(ind_lower, ind_upper)
	coeff[indexes] = 0
	return coeff

def compute_offsets(filesizes, headeroffset):
	offsets = []
	offset = headeroffset
	for size in filesizes:
		offsets.append(offset)
		offset += size
	offsets.append(offset)
	#assert(sys.getsizeof(offset) == 28)		#size of int32
	#print(f"compute_offsets {offset}")
	#print(f" compute_offsets, sizeof(offset) = {sys.getsizeof(offset)}")
	return offsets

#desigining on-disk dataset for our data.
HeaderDesc = namedtuple("images_header", "num_files, offsets, labels, zipped, colorspace")
ImageDesc = dataset_base.ImageDesc

kColorSpaces = [
	"rgb",
	"grayscale",
	"lab",
	"lum_lab",
]
def colorspace2code(colorspace: str):
	return kColorSpaces.index(colorspace)

class BigChunk(dataset_base.DataSet):	#this is compatible with torch.utils.data.Dataset
	kHeaderFmt = '>IIIII'
	kSizeofHeader = struct.calcsize(kHeaderFmt)
	kOffsetFmt = '>%sq'
	kLabelFmt  = '>%si'

	def __init__(self, 
		filename: str, mode="rb", 
		kOpen=True, 
		kPickle=False,
		colorspace="grayscale",
	):
		#print(f"BigFile")
		super().__init__(name=filename, colorspace=colorspace)
		self.numentries = 0
		self.filename = filename
		self.file = None
		self.pickle = kPickle
	
		if kOpen:	#Note: we do not call openfile() when BigChunk has to be marshalled across processes
			self.openfile(filename, mode)

	def start(self):
		""" get ourself prepared to be used as a dataset """
		if not self.isopen:
			self.openfile()
		return

	def openfile(self, filename=None, mode="rb") -> bool:
		self.filename = self.filename if filename is None else filename

		if (not self.isopen) and (mode is not None):
			self.file = open(self.filename, mode)
			fin = self.file

			if (mode == "rb"):
				self._header = self.read_header(fin)
				self.num_files, self.offset_add, self.labels_add, self.zipped, colorspace = self.header
				self._colorspace = kColorSpaces[colorspace]
				self.offsets = self.read_offsets(fin, self.offset_add, self.num_files)
				self.labels = self.read_labels(fin, self.labels_add, self.num_files)
				self.numentries = self.num_files
				#print(f"BigChunk.openfile(zipped={self.zipped})")
		return self.isopen

	def closefile(self):
		if (self.isopen):
			self.file.close()
			self.file = None
			self._header = None 	#tuple(self.header)
	
	def finalize(self):
		self.closefile()

	@property
	def isopen(self):
		return not (self.file is None)

	@property
	def header(self) -> HeaderDesc:
		return self._header

	def __len__(self):
		return self.numentries
	
	def __getitem__(self, index):
		#assigning the label of the current example.
		fin = self.file
		offsets = self.offsets
		label = self.labels[index]

		#getting the current example from the merged file.
		data = self.read_file(fin, offsets[index], offsets[index+1])
		
		stream = BytesIO()	#reading to stream. 
		stream.write(data)
		stream.seek(0)
		
		#reading into coeffs.
		if self.zipped:
			with gzip.GzipFile(fileobj = stream, mode = 'rb') as f:
				coeffs = np.load(f, allow_pickle=self.pickle)
		else:
			coeffs = np.load(stream, allow_pickle=self.pickle)
		return ImageDesc(coeffs, label)	#return a tuple instead of a dict()

	def __str__(self):
		return f"BigChunk({self.filename})"

	def getlabel(self, index):
		return self.labels[index]
		
	def get_blob(self, index):
		""" returns the compressed (blob, label) """
		fin = self.file
		offsets = self.offsets
		label = self.labels[index]

		#getting the current example from the merged file.
		data = self.read_file(fin, offsets[index], offsets[index+1])

		return ImageDesc(data, label)	#return a tuple instead of a dict()

	def write_header(self, f, size, offset, label, zipped=1, colorspace=0):
		f.seek(0)
		#print(f"write_header size={size}, offset={offset}, label={label}, zipped={zipped}")
		packed_header = struct.pack(self.kHeaderFmt, size, offset, label, zipped, colorspace)
		f.write(packed_header)
		return f.tell()

	def read_header(self, f):
		f.seek(0)
		header = struct.unpack(self.kHeaderFmt, f.read(struct.calcsize(self.kHeaderFmt)))
		#print(f"read_header {header}")
		images_header = HeaderDesc(*header)
		return images_header
	
	def write_data(self, f, data):
		cnt = f.write(data)
		return cnt
	
	def read_file(self, fin, offset, size):
		fin.seek(offset)
		return fin.read(size-offset)
		
	def  write_offsets(self, fout, filesizes: list):
		offsets = compute_offsets(filesizes, self.kSizeofHeader)
		self.numentries = len(filesizes)
		pos = fout.tell()
		offsets = struct.pack(self.kOffsetFmt % len(offsets), *offsets)
		cnt = fout.write(offsets)
		#print(f"write_offsets: len(offsets) {len(offsets)}, cnt {cnt}, pos {pos}")
		return pos
	
	def read_offsets(self, fin, offset_add, arr_len, kLogging=False):
		fin.seek(offset_add)
		bytestoread = self.labels_add - self.offset_add
		if kLogging:
			print(f"offset add: {offset_add}, arr len: {arr_len}, bytestoread {bytestoread}")
		rawoffsets = fin.read(bytestoread)
		if kLogging:
			print(f"rawoffsets: {len(rawoffsets)}")
		offsets = struct.unpack(self.kOffsetFmt %(arr_len+1), rawoffsets)
		return offsets
	
	def write_labels(self, fout, labels):
		pos = fout.tell()
		#print(f"write_labels {pos}, {labels[0:16]}")
		labels = struct.pack(self.kLabelFmt % len(labels), *labels)
		cnt = fout.write(labels)
		return pos
	
	def read_labels(self, fin, labels_add, arr_len):
		labels = None
		if labels_add != -1:
			fin.seek(labels_add)
			labels = fin.read()
			labels = struct.unpack(self.kLabelFmt % arr_len, labels)
			#print(f"read_labels {labels_add}, {labels[0:16]}")
		return np.asarray(labels)
	
	@classmethod
	def sizes2offsets(cls, sizes):
		""" convert a table of blob sizes to offsets """
		return compute_offsets(sizes, cls.kSizeofHeader)		

	@staticmethod
	def offsets2sizes(offsets):
		### convert a table of offsets into sizes """
		return tuple(offsets[i+1]-offsets[i] for i in range(len(offsets)-1))

def checkChunks(chunks):
	sizes = []
	for chunk in chunks:
		start, end, shfilename, *_ = chunk	
		sizes.append(end-start)
	cnts = Counter(sizes[0:-1])

	return (len(cnts) == 1), list(cnts.keys())[0]		#same size for all chunks	
#end of BigChunk

class Big1Chunk(BigChunk):	#this is compatible with torch.utils.data.Dataset
	""" A 1 chunk version of BigChunk where the images and labels each is saved as 1 blob """
	def __init__(self, 
		filename: str, mode="rb", 
		kOpen=True, 
		kPickle=False,
		colorspace="grayscale",
	):	
		#print(f"Big1Chunk ", end='')
		super().__init__(filename, mode, kOpen, kPickle, colorspace)

	def openfile(self, filename=None, mode="rb"):
		super().openfile(filename, mode)
		
		if self.isopen and (mode == "rb"):
			#print(f"openfile {self.num_files}, {self.offset_add}, {self.labels_add}, offsets={self.offsets}")
			self.read_images(self.offset_add, self.offsets[1]-self.offsets[0])
			self.closefile()

	def write_offsets(self, fout, filesizes: list):
		pos = super().write_offsets(fout, filesizes)
		#print(f"write_offsets filesizes=[{len(filesizes)}], pos={pos}")
		return pos
	
	def read_offsets(self, fin, offset_add, arr_len, kLogging=False):
		offsets = super().read_offsets(fin, offset_add, 1, kLogging)
		#print(f"read_offsets {offset_add}, {arr_len}, offsets={offsets}")
		return offsets

	def read_images(self, offset, sizeB):
		#tic1 = time.time()
		fin = self.file
		offsets = self.offsets
		#print(f"read_images offset={offsets[0]}, sizeB={sizeB}")
		data = self.read_file(fin, offsets[0], offsets[0]+sizeB)

		stream = BytesIO()
		stream.write(data)
		stream.seek(0)
		self.images = np.load(stream, allow_pickle=self.pickle)
		#toc1 = testutil.time_spent(tic1, f"read_images ", count=1)
		#print(f" readimages, images.shape {self.images.shape}")

	def __getitem__(self, index):
		coeffs = self.images[index]
		label = self.labels[index]
		return ImageDesc(coeffs, label)

	def __str__(self):
		return f"Big1Chunk({self.filename})"


class MultichunkDataset(BigChunk):	#this is compatible with torch.utils.data.Dataset
	""" multi-chunk BigFile """
	kChunksFmt = '>%sp'		#filenames for the BigChunks

	def __init__(self, filename, mode="rb", kOpen=True):
		self.chunksize = 1
		self.numentries = 0
		super().__init__(filename, mode, kOpen)
		#self.chunks = [self]	#default is 1 chunk 

	def openfile(self, filename=None, mode="rb"):
		self.filename = self.filename if filename is None else filename
		#print(f"MultichunkDataset.openfile({self.filename})")
		self.mode = mode
		self.ourfolder = os.path.dirname(self.filename) + '/'
		#print(f"MultichunkDataset.openfile({filename}), dirname={self.ourfolder}")

		if (not self.isopen) and (mode is not None):
			self.file = open(self.filename, mode)
			fin = self.file

			if (mode == "rb"):
				header = self.read_header(fin)
				#print(f" header {header}")
				chunks = self.readChunks(header.offsets)
				self.fromChunks(chunks, self.numentries)
			else:
				self.write_header(fin, 0, 0, 0)	
		return self.isopen
			
	def __getitem__(self, index):
		""" use index to select a chunk using // and % """
		cidx, remain = divmod(index, self.chunksize)
		ourchunk = self.chunks[cidx]
		return ourchunk.__getitem__(remain)

	def __str__(self):
		return f"MultichunkDataset({self.filename})"

	def __add__(self, rhs):
		#print(f"chunksize {self.chunksize}, {rhs.chunksize}")
		assert(self.chunksize == rhs.chunksize)
		self.chunks.extend(rhs.chunks)
		self.numentries += rhs.numentries
		return self
		
	def __iter__(self):
		""" return a new iterator """
		self.next = 0
		return self

	def __next__(self):
		""" main iteration interface """
		index = self.next
		if index < self.numentries:
			self.next += 1
			return self.__getitem__(index)
		else:
			raise StopIteration
		
	def finalize(self, chunks):
		""" chunks is the loaded shfile.dat - a list of tuples (start, end, filename) """
		if "w" in self.mode:
			chunkspos, endpos = self.writeChunks(chunks)
			self.write_header(self.file, self.numentries, chunkspos, endpos)
		self.closefile()
	
	def fromChunks(self, chunks, numentries):
		""" build our chunk table from a list of chunks coming from xpform """
		_, chunksize = checkChunks(chunks)

		ourchunks = []
		chunkoffsets = []
		for chunk in chunks:
			start, end, shfilename, *_ = chunk
			ourchunks.append(BigChunk(self.ourfolder + shfilename))
			#chunkoffsets.append(start)
			#print(f"fromChunks {end} {numentries}")
		#assert(numentries == end)

		self.chunksize = chunksize
		self.chunks = ourchunks
		self.numentries = end

	def writeChunks(self, chunks):
		""" chunks is the loaded shfile.dat - a list of tuples (start, end, filename) """
		fout = self.file
		pos = fout.tell()
		#print(f"writeChunks {chunks}")
		#1: create a datastream of bytes:
		datastream = BytesIO()
		pickle.dump(chunks, datastream)
		pickled = datastream.getvalue() 	#this is the pickled chunks
		fout.write(pickled)
		endpos = fout.tell()
		return pos, endpos

	def readChunks(self, offset_add, kLogging=False):
		self.file.seek(offset_add)
		chunks = pickle.load(self.file)
		#print(f"readChunks {chunks}")
		return chunks
#end of MultichunkDataset

class DiskShearletDataset(MultichunkDataset):	#this is compatible with torch.utils.data.Dataset
	""" For backwards compatible support """
	def __init__(self, filename, mode="rb", kOpen=True):
		super().__init__(filename, mode, kOpen)

	def __str__(self):
		return f"DiskShearletDataset({self.filename})"


def reset(stream):
	stream.truncate(0)
	stream.seek(0)

def readBlobs(bigfile):
	### TODO: make this a staticmethod """
	fin = bigfile.file
	numentries = bigfile.numentries
	offsets = bigfile.offsets
	fin.seek(offsets[0])
	nbytes = offsets[numentries] - offsets[0]
	blobs = fin.read(nbytes)
	return nbytes, blobs

def verify1(i, item1, bigfile, shearlets, threshold):
	item2 = bigfile[i]
	label1 = item1[1]
	label2 = item2.label
	sh1 = threshold(shearlets[i])
	sh2 = item2.coeffs

	result = np.array_equal(sh1, sh2)
	result &= (label1 == label2)
	return result

def verifyBigFile(bigfile, dataset, xformed, threshold=opt_threshold):
	""" verify 'mergedfile' against raw 'xformed' """
	img1 = (xformed[1], dataset[1][1])
	img11 = bigfile[1]

	print("verifying...")
	tic1 = time.process_time()
	result = True
	for i, item1 in enumerate(dataset):
		result &= verify1(i, item1, bigfile, xformed, threshold)
	assert(result)

	time_spent(tic1, "verifyBigFile()", count=1)

	return result
