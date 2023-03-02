from pathlib import Path, PurePosixPath
import multiprocessing
import os, sys, time

from mk_mlutils import projconfig
import mk_mlutils.dataset.dataset_base as dataset_base
import mk_mlutils.dataset.fashion as fashion
import mk_mlutils.utils.torchutils as torchutils
import mk_mlutils.mp.mppool as mppool
import mk_mlutils.mp.mpxform as mpxform


def xformdataset_serial(
	xform,
	dataset:dataset_base.DataSet,
	work:tuple		#(start, end)
) -> list:
	""" apply the Shearlet xform defined by 'sh_sys' to 'dataset' """
	print(f"xformdataset_serial {work}..")
	assert(type(dataset))

	#shearlets = []
	labels = []
	
	start, end = work
	for i in range(0, end-start):
		item = dataset[i]
		img, label = item

		#coeffs = sh_xform.xform(item)
		#shearlets.append(coeffs.cpu().numpy())
		labels.append(label)

	return labels

class Worker(multiprocessing.Process):
	instanceId = 0

	def __init__(self, queue, send_end, workerargs):
		""" This will be executed in the parent process """
		print(f"Worker() pid {os.getpid()}", flush=True)	#Note: this will be parent's pid		
	
		super().__init__()
		self.queue = queue
		self.send_end = send_end
		self.workerargs = workerargs
		self.instanceId = Worker.instanceId
		Worker.instanceId += 1

	def onceInit(self, kCUDA=False):
		""" per-worker|per-core once Init - we are in the worker process """
		print(f" Worker.onceInit: pid {os.getpid()}", flush=True, end='')

		kCUDA = self.workerargs.get('CUDA', False)
		self.device = torchutils.onceInit(kCUDA=kCUDA)

		self.dataset = prepWorker(self.workerargs)
		print(f" self.dataset {self.dataset}")

	@classmethod
	def qNotDone(cls, work):	
		return work is not mppool.MPPool.kSTOP_VALUE

	def oneChunk(self, xform, work:tuple) -> tuple:
		""" process 1 chunk of work """	
		start, end = work
		dbchunk = dataset_base.CustomDatasetSlice(self.dataset, ourslice=(start, end-start))
		#shearlets = xformdataset_serial(xform, dbchunk, work)
	
		#do some processing on 'dbchunk'
		xformed = xformdataset_serial(xform, dbchunk, work)

		return (work, xformed)

	@staticmethod
	def checkChunkOutput(output):
		result = (type(output) is tuple)		
		result &= len(output) == 2
		assert(result)
		return result

	def run(self):
		""" the multiprocessing.Process.run()
			We pull items off the shared queue until we see 'kSTOP_VALUE'
		"""
		q = self.queue
		workerargs = self.workerargs
#		shfactory = workerargs['shfactory'] 
		kCUDA = workerargs['CUDA']

		#1: per-core/worker once init:
		self.onceInit(kCUDA=kCUDA)

#		sh_spec, xform_factory = shfactory
#		sh_xform = xform_factory(sh_spec)
#		sh_xform.start(self.device)
#		sh_sys = sh_xform.shearletSystem
		#shearlet_spec = sh_xform.sh_spec

		results = []
		#2: persistent thread
		while True:
			work = q.get()

			if Worker.qNotDone(work):		#are we all done?
				print(f" work {work}") 

				#3: record the results in a tuple to be returned to caller
				output = self.oneChunk(None, work)
				self.checkChunkOutput(output)
				results.append(output)					
			else:
				result_tuple = (self.instanceId, results)
				self.send_end.send(result_tuple)
				results = []
				print("  poison-pill")
				break	#poison-pill
#end class Worker

class XformPool(mppool.MPPool):	
	def makeWorker(self, 
		queue:multiprocessing.Queue, 
		send_end:multiprocessing.Pipe, 
		workerargs
	):
		""" create one Worker """
		return Worker(queue, send_end, workerargs)	#this will start running immediately

	def finalize(self, kPersist:bool):
		results = super().finalize(kPersist)
		#remove the work tuples to make a final list of outputs 
		finallist = []
		[finallist.extend(lst[1]) for lst in results]
		return finallist

	def verify(self, results):
		total = 0
		#1: loop through workers
		for result in results:
			instid, output = result

			#2: loop through per-worker output list
			count = 0
			for entry in output:
				work, shearlets = entry
				count += work[1] - work[0]
				assert((work[1] - work[0]) == len(shearlets))
			total += count
		assert(total == self.numentries)
		return total == self.numentries	
#end class XformPool

def prepWorker(workerargs:dict):
	full_dataset = workerargs['full_dataset']
	full_dataset, test, validateset, *_ = fashion.load_fashion(full_dataset, validate=.3)
	return full_dataset


if __name__ == '__main__':
	torchutils.onceInit(kCUDA=True, cudadevice='cuda:1')

	print(f"{projconfig.getRepoRoot()=}")
	print(f"{projconfig.getDataFolder()=}")

	train, test, validateset, *_ = fashion.load_fashion('train', validate=.3)
	print(f"{len(train)=}, {len(test)=}, {len(validateset)=}")

	workerargs = {
		'chunkfactor': 	3,		#scheduleWork() use this to divide work finer than n_jobs 
		'full_dataset': 'train',
								#for better load-balance
		'CUDA'		 : 	False,
	}
	pool = XformPool(poolsize=2)
	pool.doit(workerargs, len(train), kPersist=False)

	xformed_results = pool.results
	print(f"{len(xformed_results)=}")

	for i, item in enumerate(train):
		assert(xformed_results[i] == item.label)

	#explicit invoke the destructor to free resources	
	del pool

		