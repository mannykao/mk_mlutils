# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""

import re
from collections import namedtuple
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Union, Optional

from mk_mlutils import projconfig
import mk_mlutils.dataset.datasetutils as datasetutils
import mk_mlutils.dataset.fashion as fashion

#from ..pipeline import loadMNIST, augmentation, dbaugmentations, trainutils

kRepoRoot="mk_mlutils/src/mk_mlutils"


if __name__ == '__main__':
	projconfig.setRepoRoot(kRepoRoot, __file__)

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
