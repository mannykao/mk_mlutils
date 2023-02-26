# test importing every file from mk_mlutils
try:
	import mk_mlutils
	import mk_mlutils.dataset
	import mk_mlutils.dataset.cifar10 as cifar10
	import mk_mlutils.dataset.dataset_base as dataset_base
	import mk_mlutils.dataset.datasetutils as datasetutils
	import mk_mlutils.dataset.fashion as fashion

	import mk_mlutils.pipeline.augmentation as augmentation

except Exception as e:
	print(f"failed to import {e} from mk_mlutils")
else:
	print("imported all files listed from mk_mlutils")

print(dir(mk_mlutils))
