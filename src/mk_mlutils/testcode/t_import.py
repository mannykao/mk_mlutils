# -*- coding: utf-8 -*-
"""
Title: test import of mk_mlutils 
	
Created on Wed Feb 8 7:01:29 2023

@author: Manny Ko & Ujjawal.K.Panchal
"""

try:
	import mk_mlutils as mk_mlutils
except:
	print("Failed to import package 'mk_mlutils'")
else:
	print("Succeeded importing 'mk_mlutils'")

try:
#	import mk_mlutils.augmentation as augmentation			#has 'cplx'
	import mk_mlutils.pipeline.batch as batch
	import mk_mlutils.pipeline.BigFile as BigFile
	import mk_mlutils.pipeline.BigFileBuilder as BigFileBuilder
	import mk_mlutils.pipeline.bigfilebuildermp as bigfilebuildermp
	import mk_mlutils.pipeline.ImagePatcher as ImagePatcher
#	import mk_mlutils.pipeline.loadCIFAR as loadCIFAR		#use augmentation
	import mk_mlutils.pipeline.loadMNIST as loadMNIST
	import mk_mlutils.pipeline.logutils as logutils
	import mk_mlutils.pipeline.mixup as mixup
	import mk_mlutils.pipeline.patching as patching
#	import mk_mlutils.pipeline.modelstats as modelstats		#has 'cplx'
	import mk_mlutils.pipeline.roc as roc
	import mk_mlutils.pipeline.torchbatch as torchbatch
#	import mk_mlutils.pipeline.trainutils as trainutils		#has CoShREM etc.

	import mk_mlutils.utils.ourlogger as ourlogger
	import mk_mlutils.utils.torchutils as torchutils
#	import mk_mlutils.utils.trace as trace
except:
	print("Failed to import package 'mk_mlutils.patching'")
else:
	print("Succeeded importing 'mk_mlutils.patching'")

print(f"{dir(mk_mlutils)=}")

#2: check to see if the modules are imported successfully
#print(f"{dir(augmentation)=}")
print(f"{dir(BigFile)=}")
print(f"{dir(BigFileBuilder)=}")
print(f"{dir(bigfilebuildermp)=}")
#print(f"{dir(loadCIFAR)=}")
print(f"{dir(loadMNIST)=}")
print(f"{dir(ImagePatcher)=}")
print(f"{dir(logutils)=}")
print(f"{dir(mixup)=}")
print(f"{dir(roc)=}")
print(f"{dir(torchbatch)=}")

print(f"{dir(ourlogger)=}")
print(f"{dir(torchutils)=}")

#print(f"{dir(trainutils)=}")

