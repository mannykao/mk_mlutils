# Pipeline

Contains all code relevant to developing an efficient pipeline for Complex Valued ShearNet.
The most prominent module is 'batch.py' which contains our batch builders.

# Content

1. [augmentation](./augmentation.py): Comprises of many types of transforms that can be performed on data. *Shearlet Transform* which is essential to this project is one of them.

2. [batch](./batch.py): Comprises of all utilities for *multiprocessing and asynchronous i/o.* based batching directly from disk (several advantages over having it in RAM, **even in terms of speed in some cases**). This utility is based on a better randomization strategy than pytorch's DataLoader and is also comparitively faster (**in most cases**).

3. [BigFile](./BigFile.py): utilities for writing the dataset from RAM. to a .gzipcompressed Disk binary file. This is done because datasetwise *shearlet transform* may produce results which might be bigger > RAM on most PCs. __See Terminology [1.]__

4. [bigfilebuildermp](./bigfilebuildermp.py): Utilities to create a BigFile (__See Terminology [1.]__) in a multiprocess fashion because creating BigFile in a serial fashion might be too slow. 

5. [combine_sh](./combine_sh.py): This file contains utilities to combine multiple bigfile chunks into 1 complete bigfile.

6. [loadMNIST](./loadMNIST.py): This file contains utilities to download/process MNIST like datasets.

7. [roc](./roc.py): Utilities to plot *ROC (receiver operating characteristic)*  curves during testing.

8. [torchbatch](./torchbatch.py): This file is used to load the batch as a Pytorch tensor on a selected device of choice.

## Terminology

1. __BigFile__: A BigFile is a preprocessd/transformed and compressed dataset stored on the disk in a chunked fashion. 

