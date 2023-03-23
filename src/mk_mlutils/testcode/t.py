import numpy as np, random
import torch

from mk_mlutils import projconfig 

kUseCplx=projconfig.kUseCplx

def transpose4Np(imgList):
	if (len(imgList.shape) == 3): #for gray imgs.
			imgList = imgList[:,np.newaxis, :, :]
	elif (len(imgList.shape) == 4): #support for colored images.
		imgList = np.transpose(imgList, (0,3,1,2))
	elif (len(imgList) == 5): #support for complex colored images.
		imgList = np.transpose(imgList, (0,4,3,1,2))
	else: #more dim support for some extra computation cost....
		dims = [dim for dim in range(len(imgList.shape))]
		top_dim = dims[0]
		bottom_dims = [dim for dim in dims[1:3]]
		channel_dims = list(dims[3:])[::-1]
		new_dims = []
		new_dims.append(top_dim)
		new_dims.extend(channel_dims)
		new_dims.extend(bottom_dims)
		imgList = np.transpose(imgList, tuple(new_dims))
	return imgList

def transpose4Torch(imgList):
	if (len(imgList.shape) == 3): #for gray image.
			imgList.unsqueeze_(1)
	elif (len(imgList.shape) == 4): #support for color imgages.
		imgList = imgList.permute(0,3,1,2)
	elif (len(imgList) == 5): #support for complex colored images.
		imgList = imgList.permute(0,4,3,1,2)
	else:
		dims = [dim for dim in range(len(imgList.shape))]
		top_dim = dims[0]
		bottom_dims = [dim for dim in dims[1:3]]
		channel_dims = list(dims[3:])[::-1]
		new_dims = []
		new_dims.append(top_dim)
		new_dims.extend(channel_dims)
		new_dims.extend(bottom_dims)
		imgList = imgList.permute(*new_dims) #args to open new_dims list.
	return imgList

dispatch = {
	np.ndarray: 	transpose4Np,		#Note: Python type object us hashable
	torch.Tensor: 	transpose4Torch,	#Note: Python type object us hashable
}

if kUseCplx:
	def transpose4Cplex(imgList):	
		imgListReal = transpose4Torch(imgList.real)
		imgListImag = transpose4Torch(imgList.imag)
		imgList = cplx.Cplx(imgListReal, imgListImag)
		return imgList

#if kUseCplx:		
#	dispatch.update(cplx.Cplx: transpose4Cplex)
