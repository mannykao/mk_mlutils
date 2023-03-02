#from distutils.core import setup
import setuptools
from setuptools import find_packages

#https://caremad.io/posts/2013/07/setup-vs-requirement/

# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
# https://docs.python.org/3/distutils/setupscript.html#installing-additional-files

packages=find_packages()
print(f"{packages=}")

setuptools.setup(name='mk_mlutils',
	version='1.3',
	description='ML and Torch Utilities',
	author='Manny Ko',
	author_email='man960@hotmail.com',
	#url='https://www.python.org/sigs/distutils-sig/',
	packages=packages,
	include_package_data=True,
	package_data={'': ['bases/*.npy']},

	install_requires=[
		"torch>=1.9",		#cuda 11.1 we are using complexFloat
		"torchsummary",
		"numpy",
		"scipy",
	],
)
