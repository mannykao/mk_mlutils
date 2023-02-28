# -*- coding: utf-8 -*-
"""	
Title: End-User tutorial for using CoShREM Xform properly.
    
Created on Tue Jan 4 13:47:29 2022

@author: Ujjawal.K.Panchal
"""
from mk_mlutils.pipeline import augmentation
from mk_mlutils.dataset import fashion, cifar10
#from mk_mlutils import  torchutils


#settings.
batchsize = 256
device = "cpu"


def power_of_2(target):
	""" round to next power-of-2 """
	if target > 1:
		for i in range(1, int(target)):
			if (2 ** i >= target):
				return 2 ** i
	else:
		return 1

def xform_fashion(device:str='cpu'):
	#1.1. Get Fashion MNIST.
	fashion_train, fashion_test, fashion_validate = fashion.load_fashion(trset = "train")

	#1.2. get batch.
	batch_images, batch_labels = fashion_train[:batchsize]
	print(f"{batch_images.shape=}, {type(batch_images)=}")

	#1.3. calculate required padding (because we can only feed hxw of powers of 2 to coshremxform).
	row_size, col_size = power_of_2(batch_images.shape[1]), power_of_2(batch_images.shape[2])
	required_pad_row, required_pad_col = (row_size - batch_images.shape[1]) // 2, (col_size - batch_images.shape[2]) // 2 
	pad = augmentation.Pad([(0,0), (required_pad_row, required_pad_row), (required_pad_col,required_pad_col)])

	#1.4. xform images.
	coshrem_xform = augmentation.CoShREM(device = device,) #Note: If you want to pass in coshrem args, use CoShREMConfig class. 
	batch_images = coshrem_xform(pad(batch_images))
	print(f"xformed: {batch_images.shape=}, {type(batch_images)=}")

def xform_cifar10(device:str='cpu'):
	#2.1. Get CIFAR 10.
	cifar10_train, cifar10_test, cifar10_validate = cifar10.load_cifar(trset = "train", colorspace = "lab")

	#2.2. get batch.
	batch_images, batch_labels = cifar10_train[:batchsize]
	print(f"{batch_images.shape=}, {type(batch_images)=}")

	#2.3. calculate required padding (because we can only feed hxw of powers of 2 to coshremxform).
	row_size, col_size = power_of_2(batch_images.shape[1]), power_of_2(batch_images.shape[2])
	required_pad_row, required_pad_col = (row_size - batch_images.shape[1]) // 2, (col_size - batch_images.shape[2]) // 2 
	pad = augmentation.Pad([(0,0), (required_pad_row, required_pad_row), (required_pad_col,required_pad_col), (0,0)]) #because cifar10 is colored it requires extra channel dim.

	#2.4. xform images.
	coshrem_xform = augmentation.CoShREM(colorspace = "rgb", device = device) #Note: If you want to pass in coshrem args, use CoShREMConfig class. 
	batch_images = coshrem_xform(pad(batch_images))
	print(f"xformed: {batch_images.shape=}, {type(batch_images)=}")	

if __name__ == '__main__':
#	device = torchutils.onceInit(kCUDA=True)

	#1. Fashion MNIST
	xform_fashion(device)

	#2. CIFAR10.
	xform_cifar10(device)
