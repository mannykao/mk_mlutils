"""
Title: ImagePatcher for flexible patching.

Created on: Jan 27 2021.

Author: Ujjawal K. Panchal
"""
#custom python imports
from typing import Iterable
from functools import partial
from itertools import product

from multiprocessing import Pool, Process, current_process
import warnings

#our imports.
from patching import find_patch


class ImagePatcher():
	def __init__(self, patch_size, stride):
		self.patch_func = partial(find_patch, patch_size = patch_size)
		self.stride = stride
		self.patch_size = patch_size

	def chunkPatcher(self, chunk):
		if current_process().name == "MainProcess":
			warnings.warn("Hint: Call this using MP child process.")

		patches4chunk = []
		#1. calculate patches for each img in chunk.
		for img in chunk:
			patches4chunk.append(self.singlePatcher(img))
		#2. return patches.
		return np.array(patches4chunk)

	def num_steps_maker(self, dimvals):
		return (int((dimvals[0] - self.patch_size) / self.stride) + 1, int((dimvals[1] - self.patch_size) / self.stride) + 1)


	def step_list_maker(self, x_steps, y_steps):
		return product(range(0, x_steps * self.stride, self.stride), range(0, y_steps * self.stride, self.stride))

	def xy_list_maker(self, dimvals):
		return self.step_list_maker(*self.num_steps_maker(dimvals))

	def singlePatcher(self, img):
		#1. make steps.
		xy_list = self.xy_list_maker((img.shape[0], img.shape[1]))
		
		#2. make and return patches.
		patches = np.array([self.patch_func(xy, img) for xy in xy_list])
		return patches

def nChunker(array, n):
	"""
	Helper function: <No neccesity. Only for Unit Test. Delete as required.>
	Divide array into N chunks.
	"""
	length = len(array)
	chunkSize = length // n
	chunkSet = []
	if not chunkSize:
		chunkSet = array #potential source of improvement (if need exists).
	else:
		for i in range(0, length, chunkSize):
			if i+chunkSize >= length:
				chunkSet.append(array[i:])
			else:
				chunkSet.append(array[i:i+chunkSize])
	return chunkSet

if __name__ == "__main__":
	#import libs.
	import multiprocessing as mp
	import time, os
	import numpy as np
	from PIL import Image

	#1. declare variables.
	n_jobs = 5
	num_runs = 10
	run_times = []
	PatcherObj = ImagePatcher(10, 80)
	test_img_folder = "test-imgs/"

	#2. read images.
	imgs = []
	for img in os.listdir(test_img_folder):
		im = np.array(Image.open(f"{test_img_folder}" +img))
		imgs.append(im)

	print(f"Running ImagePatcher Unit test for {num_runs} times:")
	#run experiment for num_runs times.
	for run in range(num_runs):
		t1 = time.time()
		chunks = nChunker(imgs, n_jobs)
		result = []
		with mp.Pool(n_jobs) as pool:
			result = pool.map(PatcherObj.chunkPatcher, chunks)
	
		t2 = time.time()
		run_times.append(t2 - t1)
		print(f"\tRuntime for run #{run} = {t2 - t1} seconds.")

	run_times = np.array(run_times)
	print(f"Mean running time: {np.mean(run_times)} seconds.")
	print(f"Std. deviation : {np.std(run_times)} seconds.")