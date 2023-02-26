import os, time
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from shnetutil import coshrem_xform
from shnetutil.cplx import visual as cplxvisual

from shnetutil.coshrem_xform import ifft2, fftshift, ifftshift, sheardec2D
from shnetutil import torchutils


if __name__ == '__main__':
	from skimage import data
	from skimage.transform import resize

	size = 256
	print(type(data.coins()))
	image = resize(data.coins(), (size,size))

	plt.figure(figsize = (6,6))
	plt.axis("off")
	#plt.imshow(image, cmap = "gray")
	#plt.show()

	# Relevant parameters for our Shearlet system
	rows, cols = image.shape
	sh_spec = coshrem_xform.ksh_spec.copy()
	sh_spec['rows'] = rows
	sh_spec['cols'] = cols
	print(f"sh_spec: {sh_spec}")

	# Generating the shearlet system with pyCoShRem
	device = torchutils.onceInit(kCUDA=True)
	coshxform = coshrem_xform.CoShXform(sh_spec)
	coshxform.start(device)
	shearlets, shearletIdxs, RMS, dualFrameWeights = coshxform.shearletSystem

	# The shearlets have 56 slices
	print(f"shearlets.shape {shearlets.shape}")

	j = 3
	shearlet = shearlets[:,:,j]

	qUpper = np.percentile(np.absolute(shearlet), 98)
	#cplxvisual.complexImageShow(shearlet/qUpper)
	#plt.show()

	# ### Visualizing the shearlets in spatial domain
	shearlet_space = fftshift(ifft2(ifftshift(shearlet)))

	qUpper = np.percentile(np.absolute(shearlet_space), 98)
	#cplxvisual.complexImageShow(shearlet_space/qUpper)
	#plt.show()

	#3: Classical way to compute the coefficients
	t = time.time()
	coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time: sheardec2D() {time.time()-t:3f}ms")

	#3.1: The coefficients are
	j = 10
	coeff = coeffs[:,:,j]
	shearlet_space = fftshift(ifft2(ifftshift(shearlets[:,:,j])))

	qUpper = np.percentile(np.absolute(coeff), 98)
	#cplxvisual.complexImageShow(coeff/qUpper)

	#3.2: Coming from the filter
	qUpper = np.percentile(np.absolute(shearlet_space), 98)
	#cplxvisual.complexImageShow(shearlet_space/qUpper)
	#plt.show()

	#4: use pytorch 
	# In order to make pytorch deal with the imaginary and real part we need to separate them in two different arrays
	shearlets_complex = coshxform.shearlets_complex
	torch_shearlets	  = coshxform.torch_shearlets
	device = coshxform.device

	# CUDA for PyTorch
	if False:
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0")
		torch.backends.cudnn.enabled = True

		torch_shearlets = torch.tensor(shearlets_complex[np.newaxis, :,:,:,:]).to(device)

		# We need to do the same for the image
		image_complex = np.concatenate((image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 2)
		torch_image = torch.tensor(image_complex[np.newaxis, :,:,:]).to(device)

		t = time.time()
		torch_coeffs = torchsheardec2D(torch_image, torch_shearlets);
		print(f"Elapsed time torchsheardec2D(): {time.time()-t:3f}ms")

	torch_coeffs = coshxform.xform((image, None))
#	torch_coeffs = coshxform.xform2(image)
	
	#Batch xform trials.
	batch_size = 64
	t1_batch = time.time()
	batch = np.repeat(image[np.newaxis,:,:], batch_size, axis = 0)
	print(f"batch type: {type(batch)}")
	print(f"my batch img shape: {batch.shape}")
	batch_torch_coeffs = coshxform.batch_xform(batch)
	print(f"Batch Xform took : {time.time() - t1_batch} s for a batch of size {batch_size}")
	
	# Comparison with the other for one batch
	t = time.time()
	coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time sheardec2D(): {time.time()-t:3f}ms")

	# ### Visualizing the coefficients produced by torch
	coeffs_numpy = torch_coeffs.detach().cpu().numpy()


	j = 10
	coeff = coeffs_numpy[:,:,j]
	qUpper = np.percentile(np.absolute(coeff), 98)
	cplxvisual.complexImageShow(coeff/qUpper)

	# Comparison with the one computed classically
	j = 10
	coeff = coeffs[:,:,j]
	qUpper = np.percentile(np.absolute(coeff), 98)
	#cplx.visual.complexImageShow(coeff/qUpper)	

	# ## Larger Batch to show advantage of using the pytorch only method
	image_complex = np.concatenate((image[:,:,np.newaxis], np.zeros(image.shape)[:,:,np.newaxis]), 2)
	print(f"batch img complex shape: {image_complex.shape}")
	# Image batch done with repeating the same image 20 times
	batch_image = np.repeat(image_complex[np.newaxis,:,:,:], 20, axis=0)
	print(f"batch image shape: {batch_image.shape}")

	#t = time.time()
	#torch_image = torch.tensor(batch_image).to(device)
	#torch_coeffs = torchsheardec2D(torch_image, torch_shearlets);
	#print(f"Elapsed time: torchsheardec2D() larger batch {time.time()-t:3f}ms")

	#  Comparison with doing the same thing outside pytorch
	t = time.time()
	for j in range(20):
		coeffs = sheardec2D(image, shearlets)
	print(f"Elapsed time: sheardec2D() larger batch {time.time()-t:3f}ms")

	#plt.show(block=True)
	#Complex Cplx testing.
	coshxform = coshrem_xform.CoShXform(sh_spec, tocplx = True)
	coshxform.start(device)
	#torch_coeffs = coshxform.xform((image, None))
	t1 = time.time()
	batch_coeffs = coshxform.batch_xform(batch)
	print(f"cplex batchxform took : {time.time() - t1} seconds.")