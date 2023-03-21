# -*- coding: utf-8 -*-
"""
Title: test import of mk_mlutils 
	
Created on Wed Feb 8 7:01:29 2023

@author: Manny Ko
"""
from pathlib import Path, PurePosixPath, PureWindowsPath, PurePath
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
from setuptools import find_packages

from mkpyutils import folderiter
import mk_mlutils.utils.importutils as importutils

kImport1=False
kImportFolder1=False
kImportFolder2=False
kSetuptools=True

try:
	import mk_mlutils as mk_mlutils
except:
	print("Failed to import package 'mk_mlutils'")
else:
	print("Succeeded importing 'mk_mlutils'")
	print(f"{  dir(mk_mlutils)=}")

modulefolder=[
	'dataset',
	'math',
	'modelling',
	'mp',
	'pipeline',
	'utils',
	'viz',
]

def importFolder(
	path:Path,					#e.g. 'mk_mlutils.dataset' 
	folder:Union[str, Path],	#file path to the source folder 
	logging:bool=True
):
	skip = {'__init__', }

	def filefilter(file:str) -> bool:
		result = folderiter.deffile_filter(file) and (file not in skip)
		return result

	print(f"{path=} {folder=}")
	files = []
	for file in list(folder.glob('*.py')):
		filename = Path(file.name).stem
		if filefilter(filename):
			if logging: print(f" {filename=}")
			files.append(filename)
	modules = importutils.importFiles(path, files, logging=logging)
	#print(modules)	
	return modules


if __name__ == "__main__":
	if kImport1:
		print(importutils.import1('mk_mlutils.utils', 'importutils'))
		print(importutils.import1('mk_mlutils.utils', 'ourlogger'))
		print(importutils.import1('mk_mlutils.utils', 'torchutils'))
		print(importutils.import1('mk_mlutils.utils', 'trace'))

	path = Path(__file__).parent.parent

	if kImportFolder1:
		importFolder('mk_mlutils.dataset', path/'dataset')

	if kImportFolder2:
		for folder in modulefolder:
			print(f"import all from 'mk_mlutils.{folder}'...")
			modules = importutils.importFolder(f'mk_mlutils.{folder}', path/folder, logging=True)
			print(modules, len(modules))
			assert(any(module is not None for module in modules))

	if kSetuptools:		
		srcroot = Path(__file__).parent.parent.parent
		print(srcroot)		

		imported = importutils.importAllPackages(srcroot, srcroot, logging=True)
		#print(imported)
		for folder in imported:
			for imp1 in folder:
				print(f"imported {imp1.__name__} as {imp1.__package__}")
			#parts = str(imp1[0].split(" from "))
			#print(parts[0])