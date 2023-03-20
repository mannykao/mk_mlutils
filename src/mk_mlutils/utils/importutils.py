# -*- coding: utf-8 -*-
"""
Title: test import of mk_mlutils 
	
Created on Sun Mar 19 7:01:29 2023

@author: Manny Ko
"""
from pathlib import Path, PurePosixPath, PureWindowsPath, PurePath
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

from mkpyutils import folderiter


#https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
def import1(path:str, module:str, logging=True) -> object:
	#'from os import path as imported'
	imported = None
	try:
		imported = getattr(__import__(path, fromlist=[module]), module)
	except:
		print(f"** Failed to import package '{path}.{module}'")
	else:
		if logging:
			print(f"Succeeded importing '{path}.{module}'")
			print(f"  {dir(imported)}")

	return imported

def importFiles(path:Union[str, Path], modules:list, logging=True) -> List[object]:
	imports = []
	for module in modules:
		imports.append(import1(path, module, logging=logging))
	return imports	

def importFolder(path:Path, folder:Union[str, Path], logging:bool=True):
	skip = {'__init__', }

	def filefilter(file:str) -> bool:
		result = folderiter.deffile_filter(file) and (file not in skip)
		return result

	if logging: print(f"{folder}")
	files = []
	for file in list(folder.glob('*.py')):
		filename = Path(file.name).stem
		if filefilter(filename):
			if logging: print(f" {filename=}")
			files.append(filename)
	modules = importFiles(path, files, logging=logging)
	#print(modules)	
	return modules


if __name__ == "__main__":
	print(import1('mk_mlutils.utils', 'ourlogger'))
	print(import1('mk_mlutils.utils', 'torchutils'))
	print(import1('mk_mlutils.utils', 'trace'))
