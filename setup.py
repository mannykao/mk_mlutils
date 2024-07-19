#from distutils.core import setup
import setuptools
from setuptools import find_packages

#https://caremad.io/posts/2013/07/setup-vs-requirement/

# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
# https://docs.python.org/3/distutils/setupscript.html#installing-additional-files

#setup.py vs setup.cfg in Python:
# https://towardsdatascience.com/setuptools-python-571e7d5500f2#:~:text=be%20more%20appropriate.-,The%20setup.,as%20the%20command%20line%20interface.
#requirements.txt vs setup.py in Python:
# https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af

	
packages=find_packages(where='src')
print(f"{packages=}")

requirements=[
	"numpy",
	"matplotlib",
	"pillow",			#PIL
	"pydantic",
	"scipy",
	"scikit-learn",
	"scikit-image",
	"funcsigs",
	"torch==1.13.1",	#temporarily pinned to this version until we sort it out
	"torchvision==0.14.1",
	"torchsummary", 
]
requirements = []

setuptools.setup(name='mk_mlutils',
	version='1.6',		#1.6 supports 'mldatasets', <= 1.5 supports 'datasets'
	description='ML and PyTorch Utilies',
	author='Manny Ko',
	author_email='man960@hotmail.com',
	#url='https://www.python.org/sigs/distutils-sig/',
	packages=packages,
	package_dir={"": 'src'},	#remove 'src' otherwise we need to do 'import src.mk_mlutils'
	include_package_data=True,
	package_data={'': ['bases/*.npy']},
	python_requires='>=3.8',
	install_requires=requirements,
)
