# -*- coding: utf-8 -*-
"""

Title: Project folder configuration manager. To have a robust means for code within 
	   our package as well as client code to locate shared datasets and other folders
	   - e.g. output/log folders, test result folders etc.
	
Created on Wed Mar 1 17:44:29 2023

@author: Ujjawal.K.Panchal & Manny Ko

"""
import re
from typing import List, Tuple, Optional, Callable
from pathlib import Path, PurePosixPath

from mkpyutils import dirutils

#
# globals configuration for the root of our repo.
# anchor for all other files especially datasets, external packages (whl) etc.
#
kRepoRoot="mk_mlutils"
#from kRepoRoot to the sources
kToSrcRoot="src/mk_mlutils"	#kRepoRoot/kToSrcRoot = "mk_mlutils/src/mk_mlutils"

#ifdef configuration flags:
kUseCplx=False		#enable cplx and CoShRem dependent code - mck
kStandAlone=False 	#control repo using our own logic instead of datasets.projconfig


def enable(**flags):
	global kUseCplx

	print(f"mk_mlutils.projconfig.enable({flags}")

	for k, v in flags.items():
		if k == 'kUseCplx':		#TODO: remove hardcoding
			kUseCplx = v

def getRefFile() -> Path:
	return Path(__file__)

def extractRepoRoot(reffile:str=__file__, reporoot:str=kRepoRoot) -> Path:
	parts = Path(reffile).parts

	ourroot = Path("")
	if (reporoot in parts):
#		print(f"found {reporoot} in {reffile}")
		for part in parts:

			if part == reporoot:
				ourroot /=  part
				break
			else:					
				ourroot /=  part
	return ourroot

kOurRoot=extractRepoRoot(__file__, kRepoRoot) / kToSrcRoot

def setRepoRoot(repo:str, reffile):
	global kOurRepo, kOurRoot

	kOurRepo = repo

	ourpath = Path(reffile) 	#D:\Dev\SigProc\onsen\venv4sh\lib\site-packages\shnetutil\projconfig.py
	posix = PurePosixPath(ourpath)
	kOurRoot = Path(re.sub(f"/{kOurRepo}/.*$", f"/{kOurRepo}/", str(posix)))

if kStandAlone:
	def getRepoRoot(reffile=__file__) -> Path:
		""" return <srcroot>/onsen where onsen is located - e.g. '<srcroot>/onsen' 
			Assumes our venv is located directly under 'onsen' which is what setup.txt prescribe.
		"""
		return Path(kOurRoot)			#D:/Dev/ML/mk_mlutils

	def getDataFolder() -> Path:
		""" return '<srcroot>/onsen/data' """
		root = getRepoRoot()
		return root / 'data'

	def getFashionMNISTFolder() -> Path:
		datafolder = getDataFolder()
		return datafolder / 'FashionMNIST' / 'raw'

	def getMNISTFolder() -> Path:
		datafolder = getDataFolder()
		return datafolder / 'MNIST' / 'raw'

	def createFashionMNISTFolder() -> Path:
		mldatasets_root = getDataFolder()
		dirutils.mkdir(str(mldatasets_root/'FashionMNIST'))
		dirutils.mkdir(str(getFashionMNISTFolder()))

	def createCIFAR10Folder():
		mldatasets_root = getDataFolder()
		dirutils.mkdir(str(mldatasets_root / 'CIFAR10'))

	def getCIFAR10Folder() -> Path:
		mldatasets_root = getDataFolder()
		return mldatasets_root / 'CIFAR10'
else:
	from mldatasets.utils.projconfig import getRepoRoot, getDataFolder, getFashionMNISTFolder, getMNISTFolder
	#from mldatasets.utils.projconfig import createFashionMNISTFolder, createCIFAR10Folder, getCIFAR10Folder


if __name__ == "__main__":
	reporoot = getRepoRoot()
	print(f"{reporoot=}")
	fashiondir = getFashionMNISTFolder()
	print(f"{fashiondir=}")
