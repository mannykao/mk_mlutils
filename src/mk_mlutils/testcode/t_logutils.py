# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""
from mk_mlutils.pipeline import logutils

if __name__ == "__main__":
	mylogger = logutils.getLogger("test_logutils")
	logutils.setup_logger(mylogger, file_name = 'test_logutils.log', 
						 kConsole = True)

	#print(dir(mylogger))
	print(mylogger.handlers)

	#1: this will go to console
	mylogger.info("testing 1..")

	#2: this will go to log file only
	logutils.disable_console(mylogger)
	print(mylogger.handlers)

	mylogger.info("testing 2..")

	#3: this will go to console
	logutils.enable_console(mylogger)
	mylogger.info("testing 3..")

	#mylogger = logutils.getLogger("fashionDCFModels")
	#logutils.setup_logger(mylogger, file_name='logs/fashionDCFModels.log', kConsole=True)
