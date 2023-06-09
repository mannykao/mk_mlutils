# -*- coding: utf-8 -*-
"""
Title: unit test for batch.py - 
	
Created on Tues Feb 28 16:01:29 2023

@author: Manny Ko 
"""

# test importing every file from mk_mlutils
try:
	import mk_mlutils
	#dataset/
	import mk_mlutils.dataset
	import mk_mlutils.dataset.datasetutils as datasetutils
#	import mk_mlutils.dataset.fashion as fashion
	#math/
	import mk_mlutils.math.sampling as sampling
	#modelling/
	import mk_mlutils.modelling.modelfactory as modelfactory
	#mp/
	import mk_mlutils.mp.mppool as mppool
	import mk_mlutils.mp.mpxform as mpxform
	#pipeline/
	import mk_mlutils.pipeline.color.colorspace as colorspace
	import mk_mlutils.pipeline.color.pytorch_colors as pytorch_colors

	import mk_mlutils.pipeline.augmentation as augmentation
	import mk_mlutils.pipeline.batch as batch
	import mk_mlutils.pipeline.BigFile as BigFile
	import mk_mlutils.pipeline.BigFileBuilder as BigFileBuilder
	import mk_mlutils.pipeline.dbaugmentation as dbaugmentation

#	import mk_mlutils.pipeline.ImagePatcher as ImagePatcher
	import mk_mlutils.pipeline.logutils as logutils
	import mk_mlutils.pipeline.modelstats as modelstats
	import mk_mlutils.pipeline.roc as roc
	import mk_mlutils.pipeline.torchbatch as torchbatch


	#utils/
	import mk_mlutils.utils.ourlogger as ourlogger
	import mk_mlutils.utils.torchutils as torchutils
#	import mk_mlutils.utils.trace as trace

	#viz/
	import mk_mlutils.viz.plotimages as plotimages

except Exception as e:
	print(f"failed to import {e} from mk_mlutils")
else:
	print("imported all files listed from mk_mlutils")

print(dir(mk_mlutils))
