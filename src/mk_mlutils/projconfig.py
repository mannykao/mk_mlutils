# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
import re
from pathlib import Path, PurePosixPath
from mkpyutils import dirutils

kOurRepo="onsen"

def setRepoRoot(repo:str):
	kOurRepo = repo

def getRepoRoot(reffile=__file__):
	""" return <srcroot>/onsen where onsen is located - e.g. '<srcroot>/onsen' 
		Assumes our venv is located directly under 'onsen' which is what setup.txt prescribe.
	"""
	#print(f"getRepoRoot.__file__ {__file__}")
	ourpath = Path(reffile) 	#D:\Dev\SigProc\onsen\venv4sh\lib\site-packages\shnetutil\projconfig.py
	posix = PurePosixPath(ourpath)
	root = Path(re.sub(f"/{kOurRepo}/.*$", f"/{kOurRepo}/", str(posix)))
	return root			#D:\Dev\SigProc\onsen

def getDataFolder():
	""" return '<srcroot>/onsen/data' """
	root = getRepoRoot()
	return root / 'data'

def getDataFolder():
	""" return '<srcroot>/onsen/data' """
	root = getRepoRoot()
	return root / 'data'

def getMNISTFolder():
	datafolder = getDataFolder()
	return datafolder / 'MNIST' / 'raw'

def getFashionMNISTFolder():
	datafolder = getDataFolder()
	return datafolder / 'FashionMNIST' / 'raw'

def createFashionMNISTFolder():
	datasets_root = getDataFolder()
	dirutils.mkdir(str(datasets_root/'FashionMNIST'))
	dirutils.mkdir(str(getFashionMNISTFolder()))

def createCIFAR10Folder():
	datasets_root = getDataFolder()
	dirutils.mkdir(str(datasets_root / 'CIFAR10'))

def getCIFAR10Folder():
	datasets_root = getDataFolder()
	return datasets_root / 'CIFAR10'
