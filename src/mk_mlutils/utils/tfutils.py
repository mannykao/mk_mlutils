"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
from typing import Callable, List, Tuple, Optional, Union
import random, os
import numpy as np
import tensorflow as tf


def Variable(
	name,
	initializer:Union[np.ndarray, tf.Tensor, Callable, np.float32],
	dtype:tf.float32,
	shape:Union[tuple, list],
	trainable=True,
) -> tf.Tensor:
	initv = initializer 	#assume it is a Tensor or ndarray
	if type(initializer) in set([np.float32,tf.float32]):
		initv = np.full(shape=shape, fill_value=1e-2, dtype=np.float32)
	if callable(initializer):
		initv = initializer(shape=shape, dtype=dtype)
	return tf.Variable(initial_value=initv, name=name, dtype=dtype, shape=shape, trainable=trainable)

def Constant(
	name,
	initializer:Union[np.ndarray, tf.Tensor, Callable, np.float32],
	dtype:tf.float32,
	shape:Union[tuple, list],
) -> tf.Tensor:
	if type(initializer) in set([np.float32,tf.float32]):
		const_init = tf.constant_initializer(value=initializer)
		#const_init is just a functor - still need to be called
	return Variable(name=name, initializer=const_init, dtype=dtype, shape=shape, trainable=False)

def tf_version() -> str:
	return tf.__version__

def tf_Cudabuilt() -> bool:
	""" our Tensorflow binary is built with Cuda """
	#reprecated: tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
	return tf.test.is_built_with_cuda()	

def tf_hascuda() -> bool:
	""" Found valid Cuda device(s) """
	return len(tf.config.list_physical_devices('GPU')) > 0

def initSeeds(seed=1):
	print(f"initSeeds({seed})")
	random.seed(seed)
	#tf.set_random_seed(seed)		#set random seed for default/global graph
	tf.random.set_seed(seed)
	#torch.cuda.manual_seed(seed)
	np.random.seed(seed)

# https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
def set_global_determinism(seed=1):
	initSeeds(seed=seed)

	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)

def onceInit(seed=1, deterministic:bool=True):
	if deterministic:
		set_global_determinism(seed)
	else:
		initSeeds(seed)
		

if __name__ == "__main__":
	print(tf_version())
	print(f"{tf_hascuda()=}, {tf_Cudabuilt()=}")

	onceInit(1)
