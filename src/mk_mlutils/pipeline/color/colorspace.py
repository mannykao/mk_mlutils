"""
MIT License

Copyright (c) 2018 Jorge Pessoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from skimage.color import (rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

# set of supported grayscale spaces
kGrayScales = set(["grayscale", "greyscale", "lum_lab"])
# set of supported colorspaces
kColorSpaces = {
	'hsv',
	'lab',
	'rgb',
	'ycb',
	'ycbcr',
	'yuv',
	'xyz',	
}

def isGrayScale(colorspace:str) -> bool:
	return colorspace in kGrayScales

def isColor(colorspace:str) -> bool:
	return not isGrayScale(colorspace)

def nchannels(colorspace:str) -> int:
	return 1 if colorspace in kGrayScales else 3

def err(type_):
	raise NotImplementedError(f'Color space conversion {type_} not implemented yet')

def getRGBkey(input_space:str) -> str:
	""" form the key for 'input_space' -> 'rgb' """
	return input_space + '2rgb' if input_space in kColorSpaces else err(input_space)

def fromRGBkey(output_space:str) -> str:
	""" form the key for 'rgb' -> 'output_space' """
	return output_space + '2rgb' if output_space in kColorSpaces else err(output_space)

def convert(input, type:str):
	return {
		'rgb2rgb': input,
		'rgb2lab': rgb2lab(input),
		'lab2rgb': lab2rgb(input),
		'rgb2yuv': rgb2yuv(input),
		'yuv2rgb': yuv2rgb(input),
		'rgb2xyz': rgb2xyz(input),
		'xyz2rgb': xyz2rgb(input),
		'rgb2hsv': rgb2hsv(input),
		'hsv2rgb': hsv2rgb(input),
		'rgb2ycbcr': rgb2ycbcr(input),
		'ycbcr2rgb': ycbcr2rgb(input)
	}.get(type)

def convert2rgb(input, type:str):
	""" convert 'input' from colorspace='type' to rgb """
	torgb = getRGBkey(type)
	return convert(input, torgb)

def convert_from_rgb(input, outtype:str):
	fromrgb = fromRGBkey(type)
	return convert(input, fromrgb)

