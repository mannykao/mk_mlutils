# -*- coding: utf-8 -*-
"""

Title: Dataset loader and conditioner for COVID dataset
    
Created on Thurs July 6 17:44:29 2020

@author: Manny Ko & Ujjawal.K.Panchal

"""
#from collections import Counter, namedtuple
from operator import itemgetter
import numpy as np 

from mk_mlutils.dataset import dataset_base
from mk_mlutils.pipeline import augmentation


class DataSetPipeline(dataset_base.DataSet):	#this is compatible with torch.utils.data.Dataset
	""" A dataset with an input transform pipeline """
	def __init__(self, 
		name='generic',
		colorspace = "grayscale", 
		sorted=False,
		imagepipeline:augmentation.BaseXform=augmentation.NullXform(),
		labelpipeline:augmentation.BaseXform=augmentation.NullXform(),

	):
		super().__init__(name=name, colorspace=colorspace, sorted=sorted)
		self.imagepipeline = imagepipeline
		self.labelpipeline = labelpipeline

	def __getitem__(self, index) -> dataset_base.ImageDesc:
		if index >= len(self):
			return None
		image = self.imagepipeline(self.images[index])
		label = self.labelpipeline(self.labels[index])
		return dataset_base.ImageDesc(image, label)


if __name__ == "__main__":
	from mk_mlutils import projconfig
	from mk_mlutils.dataset import fashion

	print(f"{projconfig.kStandAlone=}")		#should be False - i.e. we delegate to datasets.utils.projconfig for dataset folders
	fashiondir = projconfig.getFashionMNISTFolder()
	print(fashiondir)

	fashion_train, fashion_test, validateset, *_ = fashion.load_fashion('test', validate=0.2)
	#print(fashion_train, fashion_test, validateset)


