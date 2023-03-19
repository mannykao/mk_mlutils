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

modules =[
	'dbaugmentations',	#this is needed for trainutils
	'trace',			#this is needed for trainutils
	'trainutils',
]

# import mk_mlutils.pipeline.dbaugmentations as dbaugmentations
# import mk_mlutils.utils.trace as trace
# import mk_mlutils.pipeline.trainutils as trainutils

print(importutils.import1('mk_mlutils.pipeline', 'dbaugmentations'))
print(importutils.import1('mk_mlutils.utils', 'trace'))
print(importutils.import1('mk_mlutils.pipeline', 'trainutils'))

#importutils.importFiles('mk_mlutils.pipeline', modules)
