# -*- coding: utf-8 -*-
"""
Title: Helper to setup and use 'logging'

@author: Manny Ko & Ujjawal.K.Panchal
"""
import numpy as np
import torch

#
# Mixup: https://arxiv.org/abs/1710.09412, https://github.com/facebookresearch/mixup-cifar10
# MixMatch: https://arxiv.org/abs/1905.02249, https://github.com/google-research/mixmatch
#

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda
	   Extracted from https://github.com/facebookresearch/mixup-cifar10/train.py - mck
	'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)	#Beta distribution, see p.3 of ICLR18
	else:
		lam = 1		#-ve 'alpha' => no mixup

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
