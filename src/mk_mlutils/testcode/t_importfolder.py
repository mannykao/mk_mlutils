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
kImportFolder2=True
kSetuptools=False

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

def importAllPackages(where:str, srcroot:Union[str, Path]) -> list:
	""" 	#D:/Dev/ML/mk_mlutils/src """
	packages = find_packages(where=where)
	print(f"{packages=}")
	imported = []
	for package in packages:
		if package == "mk_mlutils":
			continue
		parts = package.split(".")
		print(f"{'.'.join(parts[:-1])} {parts[-1]}")
		pkg = importutils.importFolder(package, srcroot)
		imported.append(pkg)
	return imported


if __name__ == "__main__":
	if kImport1:
		print(importutils.import1('mk_mlutils.utils', 'importutils'))
		print(importutils.import1('mk_mlutils.utils', 'ourlogger'))
		print(importutils.import1('mk_mlutils.utils', 'torchutils'))
		print(importutils.import1('mk_mlutils.utils', 'trace'))

	path = Path(__file__).parent.parent

	if kImportFolder1:
		importutils.importFolder('mk_mlutils.utils', path/'utils')

	if kImportFolder2:
		for folder in modulefolder:
			print(f"import all from 'mk_mlutils.{folder}'...")
			modules = importutils.importFolder(f'mk_mlutils.{folder}', path/folder, logging=True)
			print(modules, len(modules))
			assert(any(module is not None for module in modules))

	if kSetuptools:		
		srcroot = Path(__file__).parent.parent.parent
		print(srcroot)		

		imported = importAllPackages(str(srcroot), srcroot)
		print(imported)
