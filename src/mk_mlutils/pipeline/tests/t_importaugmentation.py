# -*- coding: utf-8 -*-
"""
Title: test import of mk_mlutils 
	
Created on Wed Feb 8 7:01:29 2023

@author: Manny Ko
"""

import mk_mlutils.utils.importutils as importutils

try:
	import mk_mlutils as mk_mlutils
except:
	print("Failed to import package 'mk_mlutils'")
else:
	print("Succeeded importing 'mk_mlutils'")
	print(f"{  dir(mk_mlutils)=}")

modules1 =[
	'augmentation',
	'batch',
	'BigFile',
	'BigFileBuilder',
	'bigfilebuildermp',
	'ImagePatcher',
	'loadCIFAR',		#use augmentation
	'loadMNIST',
	'logutils',
	'mixup',
	'patching',
	'modelstats',		#has 'cplx'
	'roc',
	'torchbatch',
	'trainutils',	#has CoShREM etc.
]

modules2 =[
	'ourlogger',
	'torchutils',
	'trace',
]

#print(importutils.import1('mk_mlutils.pipeline', 'patching'))
#print(importutils.import1('mk_mlutils.pipeline', 'loadCIFAR'))
#print(importutils.import1('mk_mlutils.pipeline', 'modelstats'))

importutils.importFiles('mk_mlutils.pipeline', modules1)
importutils.importFiles('mk_mlutils.utils', modules2)





