
#import mk_mlutils.dataset.fashion as fashion
import re
from collections import namedtuple
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Union, Optional

from mk_mlutils import projconfig
import mk_mlutils.dataset.datasetutils as datasetutils
#from ..pipeline import loadMNIST, augmentation, dbaugmentations, trainutils

kRepoRoot="mk_mlutils/src/mk_mlutils"


if __name__ == '__main__':
	projconfig.setRepoRoot(kRepoRoot, __file__)
	repoRoot = projconfig.getRepoRoot()
	print(f"{repoRoot=}")
	print(f"{projconfig.getDataFolder()=}")

	datasetdir = repoRoot/'dataset'
	print(f"{datasetdir=}")

	if True:
		fashiondir  = projconfig.getFashionMNISTFolder(),
		mnistdir	= projconfig.getMNISTFolder(),
		print(f"{fashiondir=}")
		print(f"{mnistdir=}")

