# -*- coding: utf-8 -*-
"""

Title: Testing torchvision's pretrained models.
	
Created on Thurs Apr 13 17:44:29 2023

@author: Manny Ko & Ujjawal.K.Panchal 

"""
#from pathlib import Path, PurePath
import pprint
from torchvision import models
import torch

from torchvision.models import vgg16 as _vgg16

#
#Note: this does not work inside Sublime (it needs openSSL dll) if it needs to download from torch.hub
#      Run the script in Miniconda shell the first time to DL to local cache
#Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to C:\Users\manny/.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth

#https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url


kTorchHub=False
kLocalCache=True

def download_weights(ourmodels:list) -> list:
	pass

if __name__ == '__main__':
	pp = pprint.PrettyPrinter(indent=1, width=120)
	#print(dir(models))

	checkpt_dir = torch.hub.get_dir()
	print(f"{checkpt_dir=}")
	#print(torch.hub.list())	

	#1: pre 15
	#alexnet = models.alexnet(pretrained=True)

	#2: post 15:
	if kTorchHub:
		alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
		resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
		
		pp.pprint(alexnet)
		pp.pprint(resnet50)
		pp.pprint(vgg16)

	#3: try hub.load() directly from our hub cache (must have a correct hubconf.py)
	if kLocalCache:
		alexnet = torch.hub.load(checkpt_dir, 'alexnet', source='local', weights='AlexNet_Weights.DEFAULT')
		pp.pprint(alexnet)

		resnet50 = torch.hub.load(checkpt_dir, 'resnet50', source='local', weights='ResNet50_Weights.DEFAULT')
		pp.pprint(resnet50)

		vgg16 = torch.hub.load(checkpt_dir, 'vgg16', source='local', weights='VGG16_Weights.DEFAULT')
		pp.pprint(vgg16)

	#checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
	#model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))