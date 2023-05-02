"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import tensorflow as tf

def tf_version() -> str:
	return tf.__version__

def tf_Cudabuilt() -> bool:
	""" our Tensorflow binary is built with Cuda """
	#reprecated: tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
	return tf.test.is_built_with_cuda()	

def tf_hascuda() -> bool:
	""" Found valid Cuda device(s) """
	return len(tf.config.list_physical_devices('GPU')) > 0


if __name__ == "__main__":
	print(tf_version())
	print(f"{tf_hascuda()=}, {tf_Cudabuilt()=}")
