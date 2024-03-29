# -*- coding: utf-8 -*-
"""
Title: Model Factory - 
	
Created on Fri Oct 1 7:01:29 2021

@author: Manny Ko & Ujjawal.K.Panchal
"""
from typing import List, Tuple, Union, Optional, Callable
from collections import namedtuple

from sklearn.metrics import confusion_matrix
import tqdm

import torch
import torch.nn.functional as F

from mk_mlutils import projconfig 
from mk_mlutils.utils import trace
from mk_mlutils.pipeline import batch, torchbatch


Model_Score = namedtuple("Model_Score", "cm precision recall loss")

def softmax_nll(finalactivation, target: torch.Tensor, reduction="mean") -> torch.Tensor:
	""" this behaves like F.cross_entropy but give us access to the probabilities. https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
	"""
	#F.cross_entropy(pred, target, reduction=reduction) 
	logsoftmax = F.log_softmax(finalactivation, dim=1)
	return F.nll_loss(logsoftmax, target, reduction=reduction)

def recordIncorrect(recorder, bi, rest, prediction):
	_, preds = torch.max(prediction, 1)
	wrongPred = (preds != rest)
	for i in range(len(rest)):
		if wrongPred[i]:
			recorder.log(bi, i, rest[i], prediction[i])

def model_predict_basic(model, batchbuilder, xform = None, device = "cpu") -> tuple: #TODO: Accept 2 xforms.
	"""Compute the model prediction on data from the feed."""
	dbchunk = batchbuilder.dataset
	pred, fact = [], []
	model.to(device)

	with torch.no_grad():
		model.eval()
		epoch = batchbuilder.epoch(False)

		for bi, mybatch in enumerate(epoch):
			#data, labels = mybatch
			data, labels = torchbatch.getBatchAsync(device, dbchunk, mybatch, imgXform = xform) #TODO: accept 2 xforms.

			prediction = model(data)
			pred.append(prediction)
			fact.append(labels)

	fact = torch.cat(fact, dim=0).cpu()
	return torch.cat(pred, dim=0).cpu(), fact

def model_predict(model, batchbuilder, xform = None, device = "cpu") -> tuple:
	"""Compute the model prediction on data from the feed."""
	dbchunk = batchbuilder.dataset
	pred, fact = [], []
	model.to(device)
	recorder = trace.IndexRecorder(model_id = model.__class__.__name__, 
								   id = "wrongpred")
	with torch.no_grad():	#tqdm.tqdm(dbchunk) as bar,
		model.eval()
		epoch = batchbuilder.epoch(False)

		for bi, mybatch in enumerate(epoch):
			data, labels = torchbatch.getBatchAsync(device, dbchunk, mybatch, xform = xform)

			prediction = model(data)
			pred.append(prediction)
			fact.append(labels)
			#TODO:
			recordIncorrect(recorder, bi, labels, prediction)
	recorder.save()
	fact = torch.cat(fact, dim=0).cpu() if fact else None
	return torch.cat(pred, dim=0).cpu(), fact

