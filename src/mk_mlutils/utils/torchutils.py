# -*- coding: utf-8 -*-
"""

Title: Shearlet based CNN vs. simple MLP in Fashion MNIST.
	
Created on Mon Mar 16 17:44:29 2020

@author: Ujjawal.K.Panchal & Manny Ko

"""
from pathlib import Path
import random
import numpy as np
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from typing import Union, Optional, Tuple

import torch
import torch.optim as optim
import torchsummary

from mkpyutils import dirutils

#use the right datatype for FloatTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor  = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ModelSizes = namedtuple("ModelSizes", "num_w trainable sizeB")

#import gpu.cudautils	#we are not using these yet, it has more power and control over cuda
kSnapShotDir = 'snapshots/'

kAutoDetect=False

if kAutoDetect:
	import pycuda.driver as cuda
	
	class aboutCudaDevices():
		def __init__(self):
			""" pycuda.driver """
			cuda.init()
		
		def num_devices(self):
			"""Return number of devices connected."""
			return cuda.Device.count()
		
		def devices(self):
			"""Get info on all devices connected."""
			devices = []
			num = cuda.Device.count()
			for id in range(num):
				name = cuda.Device(id).name()
				memory = cuda.Device(id).total_memory()
				devices.append((memory, name, id))
			return devices

		def preferred(self):
			""" return preferred cuda device - (<memory>, <name>, <id>) """
			#1: sort by memory size	- use better smarts when needed
			devices = sorted(self.devices(), reverse=True)
			return devices

		def mem_info(self):
			"""Get available and total memory of all devices."""
			available, total = cuda.mem_get_info()  #Note: pycuda._driver.LogicError: cuMemGetInfo failed: context is destroyed
			print("Available: %.2f GB\nTotal:     %.2f GB"%(available/1e9, total/1e9))
			
		def attributes(self, device_id=0):
			"""Get attributes of device with device Id = device_id"""
			return cuda.Device(device_id).get_attributes()
		
		def __repr__(self):
			"""Class representation as number of devices connected and about them."""
			num = cuda.Device.count()
			string = ""
			string += ("%d device(s) found:\n"%num)
			for i in range(num):
				string += ( "    %d) %s (Id: %d)\n"%((i+1),cuda.Device(i).name(),i))
				string += ("          Memory: %.2f GB\n"%(cuda.Device(i).total_memory()/1e9))
			return string

	def get_cuda(cudadevice='cuda:0'):
		""" return the best Cuda device """
		#print ('Available cuda devices ', torch.cuda.device_count())
		if cudadevice is None:
			about = aboutCudaDevices()
			devid = about.preferred()[2]
		else:		
			devid = cudadevice
		#print ('Current cuda device ', devid, torch.cuda.get_device_name(devid))
		#device = 'cuda:0'	#most of the time torch choose the right CUDA device
		return torch.device(devid)		#use this device object instead of the device string
else:
	def get_cuda(cudadevice='cuda:0'):
		""" return the best Cuda device """
		devid = cudadevice
		#print ('Current cuda device ', devid, torch.cuda.get_device_name(devid))
		#device = 'cuda:0'	#most of the time torch choose the right CUDA device
		return torch.device(devid)		#use this device object instead of the device string
#if kAutoDetect

def initSeeds(seed=1, kLogging:bool=False):
	if kLogging: print(f"initSeeds({seed})")
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)		#lmu_psmnist.py
	np.random.seed(seed)

def onceInit(kCUDA=False, cudadevice='cuda:0', seed=1):
	#print(f"onceInit {cudadevice}")
	if kCUDA and torch.cuda.is_available():
		#did user specify a specific cuda device?
		if cudadevice is None:
			device = get_cuda() 	#invoke intelligent cuda device discovery to select 'best'
		else:
			device = torch.device(cudadevice)
			torch.cuda.set_device(device)
		torch.cuda.empty_cache()
		#print(torch.cuda.get_device_name(torch.cuda.current_device()))
	else:
		device = 'cpu'
	print(f"torchutils.onceInit device = {device}")
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False		#lmu_psmnist.py
	torch.backends.cudnn.enabled = kCUDA

	initSeeds(seed)

	return device

def allocatedGPU():
	# Returns the current GPU memory usage by 
	# tensors in bytes for a given device
	return torch.cuda.memory_allocated()	

def shutdown():
	# Releases all unoccupied cached memory currently held by
	# the caching allocator so that those can be used in other
	# GPU application and visible in nvidia-smi
	torch.cuda.empty_cache()	

def createSnapshotFolder(snapshot_name:str):		
	folder = Path(snapshot_name).parent
	dirutils.mkdir(str(folder))	

kOptimState='optimizer_state_dict'

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_snapshot(epoch, net, optimizer, running_loss, snapshot_name, recipe, datasetname, trset):
	""" create a snapshot for the model's parameters and optionally the optimizer's state """
	state =	{
		'epoch':	epoch,
		'model_state_dict': net.state_dict(), 
		'loss': running_loss,
		'recipe': recipe,
		'dataset': datasetname,
		'trset': trset,
	}
	if optimizer:	#this makes the snapshot much bigger
		state.update({kOptimState: optimizer.state_dict()})

	createSnapshotFolder(snapshot_name)	
	torch.save(
		state,
		snapshot_name+'.pth'
#		snapshot_name+str(epoch+1)+'.pth'	#TODO: encode the epoch we stopped at
	)

def restore_state(model: torch.nn.Module, snapshot: dict):
	assert(type(snapshot) == dict)
	model.load_state_dict(snapshot['model_state_dict'])

#
# https://medium.com/@FrancescoZ/loading-huge-pytorch-models-with-linear-memory-consumption-449562fb2190
#

def load_snapshot(device, model, snapshot_name, optimizer=None):
	""" load a snapshot into 'device' and restore the model_state """
	try:
		snapshot = torch.load(snapshot_name+'.pth', map_location=device)
		if model:
			model.load_state_dict(snapshot['model_state_dict'])
			if optimizer:
				print(f" restore_optimizer..")
				restore_optimizer(optimizer, snapshot)
	except:
		print(f"\n** load_snapshot failed on {snapshot_name+'.pth'}")
		#assert(False)
		snapshot = None	
	return snapshot

def load1model(device, folder, snapshot_name, model, epoch=None):
	print(f" load model '{snapshot_name}', ", end='')
	snapshot = load_snapshot(device, model, snapshot_name=snapshot_name, optimizer=None)
	#state_dict = model.state_dict()
	#print(f"statedict {len(state_dict)}")
	return snapshot

def restore_optimizer(optimizer, snapshot):
	optim_state = snapshot.get(kOptimState, None)
	if optim_state:
		optimizer.load_state_dict(optim_state)

def snapshotname(modelname:str, folder:str =kSnapShotDir):
	snapshot_name = folder + modelname
	return snapshot_name

def save1model(
	name, 
	model, 
	epochs=None, 
	optimizer=None, 
	recipe=None, 
	datasetname="fashion",
	trset="test",
	klog=True
) -> str:
	snapshot_name = snapshotname(name)
	if klog:
		print(f"save model '{snapshot_name}', epoch {epochs}")
	state_dict = model.state_dict()

	save_snapshot(
		epochs, 
		model, 
		optimizer=optimizer, 
		running_loss=None,
		snapshot_name=snapshot_name, 
		recipe=recipe,
		datasetname=datasetname,
		trset=trset,
	)
	return snapshot_name
	
def load1model(
	device, 
	folder, 
	name, 
	model, 
	epoch=None,
	optimizer=None	
):
	snapshot_name = snapshotname(name, folder)
	print(f" load model '{snapshot_name}' ", end='')
	snapshot = load_snapshot(device, model, snapshot_name=snapshot_name, optimizer=optimizer)
	#state_dict = model.state_dict()
	#print(type(state_dict), len(state_dict))
	return snapshot

def loadCheckPoint(device, ckpt_dir:str, checkptname:str, logging:bool=False):
	""" Load a .pth checkpoint file """
	if Path(checkptname).suffix == "":
		checkptname += '.pth'
	ckpt_path = Path(ckpt_dir)/checkptname
	print(f"{ckpt_path=}")
	checkpoint = torch.load(str(ckpt_path), map_location=device)	

	return checkpoint

def checkpointSize(checkpoint:torch.nn.Module, details=False, logging=True) -> ModelSizes:
	""" Returns number of weights in the checkpoint """
	return dumpModelSize(checkpoint, details=details, logging=logging)

def is_cuda_device(device):
	""" predicate for 'device' being a cuda device """
	return 'cuda' in str(device)

def is_cuda(model):
	""" predicate for 'model' begin on a Cuda device """
	return next(model.parameters()).is_cuda

def log_summary(model, ip_shape, logfile):
	successful = False
	try:
		with open(logfile, 'wt') as fout:
			summary, _ = torchsummary.summary_string(model, ip_shape, device = 'cpu')
			fout.write(summary)
			sucessful = True
	except:
		print(f"Error logging :> model : {model.__class__.__name__}, ip_shape : {ip_shape}, logfile : {logfile} ")
	return successful

def choose_optimizer(optimizer, optim_params):
	""" choose the selected optimizer based on its name """    
	if(optimizer == 'adam'):
		optimizer = optim.Adam(**optim_params)
	elif(optimizer == 'adamw'):
		optimizer = optim.AdamW(**optim_params)
	elif(optimizer == 'asgd'):
		optimizer = optim.ASGD(**optim_params)
	elif(optimizer == 'adamax'):
		optimizer = optim.Adamax(**optim_params)
	elif(optimizer == 'adagrad'):
		optimizer = optim.Adagrad(**optim_params)
	elif(optimizer == 'asgd'):
		optimizer = optim.ASGD(**optim_params)
	elif(optimizer == 'rmsprop'):
		optimizer = optim.RMSprop(**optim_params)
	return optimizer

def getBatch(dataset, indices):
	print(type(indices))
	batch = np.sort(indices)

def sizeB(tensor:Union[torch.Tensor, np.ndarray]) -> int:
	""" return number of elements and size in bytes """
	if type(tensor) == torch.Tensor:
		numE = tensor.numel()
		sizeb = numE * tensor.element_size()
	else:
		numE = tensor.size
		sizeb = numE * tensor.itemsize
	return sizeb

def layers(checkpt:Union[torch.nn.Module, OrderedDict]) -> Tuple[str, torch.Tensor]:
	""" Generator for layers in 'checkpt' """
	if type(checkpt) == OrderedDict:
		for k,v in checkpt.items():
			yield k, v
	else:
		for name, param in model.named_parameters():
			yield k, v

def getLayersbyPrefix(checkpt:Union[torch.nn.Module, OrderedDict], prefix:str="") -> OrderedDict:
	""" Returns all the layers that match given prefix """
	pre_len = len(prefix)
	output = OrderedDict()
	for layer in layers(checkpt):
		name, param = layer
		if name[:pre_len] == prefix:
			#print(name)
			output.update({name: param})
	return output

def formatModelSize(ms:ModelSizes, tag:str="Total number of weights") -> str:
	return f"{tag} = {(ms.num_w/(1024*1024)):.2f}m, trainable {(ms.trainable/(1024*1024)):.2f}m, sizeB {ms.sizeB/(1024*1024):.2f}mb"

def layersStat(checkpt:Union[torch.nn.Module, OrderedDict], prefix:Union[str, list]="", klogging:bool=True) -> ModelSizes:
	""" Returns (m_w, sizeb) for layers matching prefix """
	total_num_w, total_trainable, total_sizeb = 0, 0, 0
	if klogging: print(f"layersStat('{prefix}')")

	if not isinstance(prefix, list):
		prefix = [prefix]

	for p in prefix:	
		num_w, sizeb = 0, 0
		layers = getLayersbyPrefix(checkpt, prefix=p)
		#print(f"{prefix}: ")

		for k, m in layers.items():
			num_w += m.numel()
			sizeb += sizeB(m)
		if klogging: print(f" {p}*: {num_w=}, {sizeb=}")
		total_num_w += num_w
		total_sizeb += sizeb
	total_trainable = total_num_w

	return ModelSizes(total_num_w, total_trainable, total_sizeb)

def dumpModelSize(model, details=True, logging=True) -> ModelSizes:
	""" return total number of parameters and trainables """
	total = 0
	size_b = 0
	num_trainable = 0

	if type(model == OrderedDict):
		for k, v in model.items():
			if details: print(f"{k}:, {v.shape}, {torch.numel(v)}, {sizeB(v)/1024:.1f}kb")
			total += torch.numel(v)
			size_b += sizeB(v)
		num_trainable = total 	#assuming all are trainable
	else:
		total = sum(p.numel() for p in model.parameters())
		if details:
			for name, param in model.named_parameters():
				if param.requires_grad:
					n = sum(p.numel() for p in param)
					num_trainable += n
					sizeb += sum(sizeB(p) for p in param)
					if logging:		#dump trainable parameters as percentage of total
						print(f"name: {name}, num params: {n} ({(n/total) *100 :.2f}%)")
		if logging:
			print(f"total params: {total}, ", end='')
			print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
	return ModelSizes(total, num_trainable, size_b)

def dumpLayers(model):
	for name, layer in getattr(model, "named_modules", ()):
		if name != '':	#do not dump the model itself
			print(f" {name}: {layer}")		

def modelName(model):
	modelname = getattr(model, "_modelName", '')
	return modelname if (modelname != '') else model.__class__.__name__

def getlayerByName(model, layername):
	layer = None
	for name, module in model.named_modules():
		if name == layername:
			layer = module 
	#assert(layer is not None), f"Error<!>: layername: {layername} is not present in the given model class: {model}."
	return layer

def histogram(param: torch.nn.parameter.Parameter, bins=10):
	""" TODO: move this to torchutils """
	assert(type(param) == torch.nn.parameter.Parameter)
	param_np = param.detach().cpu().numpy()
	#print(type(param_np), param_np.shape)
	hist = np.histogram(param_np, bins=bins)
	return hist
	
def getModulesByName(model, attrlist):
	""" Retrieve a list of nn.Modules inside 'model' by their names 
		The 'attrlist' are the list of properties in the model class, NOT their torch module names,
	"""
	result = []
	attrset = set(attrlist if not isinstance(attrlist, str) else [attrlist])

	if getattr(model, 'seqdict', None):
		for k, fc in model.seqdict.items():
			if k in attrset:
				result.append((k, fc))
	else:
		result = [ (attr, getattr(model, attr, None)) for attr in attrlist]
	return result
		
if False:
	import torchprofile

	def profile_mac(model, input_shape:tuple)->int:
		inputs = torch.randn(size=input_shape)
		macs = torchprofile.profile_macs(model, inputs)
		return macs


