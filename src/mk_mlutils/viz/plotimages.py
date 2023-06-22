# -*- coding: utf-8 -*-
"""
Title: unit test for batch.py - 
	
Created on Tues Feb 28 16:01:29 2023

@author: Manny Ko 
"""
#from pathlib import Path
import matplotlib.pyplot as plt

#** https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly

kRepoRoot="mk_mlutils/src/mk_mlutils"

def grid2d(
	dataset,
	columns = 5, rows = 2, 
	title:str="MNIST[0:10]"
):
	""" plot first 10 images in 2 rows	 """
	fig = plt.figure(title)
	ax = []		#for saving all subplots

	for i in range(0, columns*rows):
		img, imglbl = dataset[i] 	#result of our model on the test_set[i]
		subplot = fig.add_subplot(rows, columns, i+1)
		ax.append(subplot)
		subplot.set_title(f"{imglbl}")
		plt.imshow(img, cmap='gray')	

if __name__=='__main__':
	from mk_mlutils import projconfig
	from datasets.mnist import mnist

	projconfig.setRepoRoot(kRepoRoot, __file__)

	mnistfolder = projconfig.getMNISTFolder()
	print(mnistfolder)
	mnist_test = mnist.getdb(mnistfolder, istrain=False, kFashion=False, kTensor = False)

	grid2d(mnist_test, 5, 2, "MNIST[0:10]")

	plt.show(block=True)
	#testing
	