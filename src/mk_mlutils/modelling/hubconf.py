# -*- coding: utf-8 -*-
"""

Title: Sample 'hubconf.py' to configure a local torch snapshots/weights.
	
Created on Thurs Apr 13 17:44:29 2023

@author: Manny Ko

"""
dependencies = ['torch']
from torchvision import models
from torchvision.models.alexnet import alexnet as _alexnet
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.models.resnet import resnet50 as _resnet50
from torchvision.models import vgg16 as _vgg16

#
#https://pytorch.org/docs/stable/hub.html
#
def alexnet(weights=models.AlexNet_Weights.DEFAULT, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load latest pretrained weights
    model = _alexnet(weights=weights, **kwargs)
    return model

def resnet18(weights=models.ResNet18_Weights.DEFAULT, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load latest pretrained weights
    model = _resnet18(weights=weights, **kwargs)
    return model

def resnet50(weights=models.ResNet50_Weights.DEFAULT, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load latest pretrained weights
    model = _resnet50(weights=weights, **kwargs)
    return model

def vgg16(weights=models.VGG16_Weights.DEFAULT, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load latest pretrained weights
    model = _vgg16(weights=weights, **kwargs)
    return model

