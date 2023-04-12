"""
Sample code given in root's README.md of `semantic-segmentation-pytorch`,
has been adopted to work in our `SuperPixelSeg` repo.
---
Author: Ujjawal.K.Panchal & Manny Ko

"""
import torch
import numpy as np
from tqdm import tqdm, trange
from torchvision import transforms

from mk_mlutils.pipeline import batch
from mk_mlutils.pipeline import torchbatch

import datasets.bsd500.bsd500 as bsd500

from mk_mlutils.pipeline.augmentation import RescaleImgBatch, ImgBatchToTensor, BSDLabelDrop, OnlyToTensor

print(f"{torchbatch.kUseCplx=}")