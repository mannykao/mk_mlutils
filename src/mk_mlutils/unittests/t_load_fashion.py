# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""
#from typing import List, Tuple, Union, Optional

from datasets.fashionmnist import fashion
from mk_mlutils import projconfig

kRepoRoot="mk_mlutils/src/mk_mlutils"

#
# testing fashion.load_fashion() in standalone mode
#

if __name__ == '__main__':
	#1: explicit setting our repo root is no longer necessary, but the following line works
	#projconfig.setRepoRoot(kRepoRoot, __file__)

	repoRoot = projconfig.getRepoRoot()
	print(f"{repoRoot=}")
	print(f"{projconfig.getDataFolder()=}")

	datasetdir = repoRoot/'dataset'
	print(f"{datasetdir=}")

	fashiondir  = projconfig.getFashionMNISTFolder(),
	mnistdir	= projconfig.getMNISTFolder(),
	print(f"{fashiondir=}")
	print(f"{mnistdir=}")

	fashion_train, fashion_test, validateset, *_ = fashion.load_fashion('train', validate=0.2)
	print(f"{len(fashion_train)=}, {len(fashion_test)=}, {len(validateset)=}")
