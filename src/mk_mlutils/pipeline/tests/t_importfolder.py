# -*- coding: utf-8 -*-
"""
Title: test import of mk_mlutils 
	
Created on Wed Feb 8 7:01:29 2023

@author: Manny Ko
"""
from pathlib import Path, PurePosixPath, PureWindowsPath, PurePath
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

import mk_mlutils.utils.importutils as importutils

kImport1=False
kImportFolder1=False

try:
	import mk_mlutils as mk_mlutils
except:
	print("Failed to import package 'mk_mlutils'")
else:
	print("Succeeded importing 'mk_mlutils'")
	print(f"{  dir(mk_mlutils)=}")

modulefolder=[
#	'datasets',
	'utils',
]

def importFolder(path:Path, folder:Union[str, Path], logging:bool=True):
	if logging: print(f"{folder}")
	files = []
	for file in list(folder.glob('*')):
		files.append(Path(file.name).stem)
	modules = importutils.importFiles(path, files, logging=logging)
	#print(modules)	
	return modules


if __name__ == "__main__":
	if kImport1:
		print(importutils.import1('mk_mlutils.utils', 'importutils'))
		print(importutils.import1('mk_mlutils.utils', 'ourlogger'))
		print(importutils.import1('mk_mlutils.utils', 'torchutils'))
		print(importutils.import1('mk_mlutils.utils', 'trace'))

	path = Path(__file__).parent.parent.parent

	if kImportFolder1:
		importutils.importFolder('mk_mlutils.utils', path/'utils')

	for folder in modulefolder:
		modules = importutils.importFolder(f'mk_mlutils.{folder}', path/folder, logging=True)
		#print(modules)
