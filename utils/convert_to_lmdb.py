#https://gist.github.com/longjon/ac410cad48a088710872#file-readme-md
 
#convert_to_lmdb.py
import caffe
import lmdb
from random import shuffle
import scipy.io as sio
import scipy
import sys
import os
import shutil
import random

from PIL import Image

import numpy as np

import glob

class CArgs(object):
	pass

#args = CArgs()



class Resizer:
	def __init__(self, size, box_size, nopadding=False, RGB_pad_value=None, flow_pad_value=128):
		self.size = size
		self.box_size = box_size
		self.nopadding = nopadding
		self.RGB_pad_value = RGB_pad_value
		self.flow_pad_value = flow_pad_value
		
	def padarray(self, im, padval):
		if self.nopadding:
			return im
		
		box_size = self.box_size
		box_size = (max(im.shape[0], im.shape[1]), max(im.shape[0], im.shape[1])) if box_size==None else box_size
		
		pad_size = (box_size[0]-im.shape[0], box_size[1]-im.shape[1])
		if im.ndim==2: #flow or label
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = padval)
		else: #RGB
			arrays = [np.pad(im[:,:,ch], pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = padval[ch]) for ch in range(im.shape[2])]
			pad_im = np.stack(arrays, axis=2)
		return pad_im

class ImageResizer(Resizer):
	def resize(self, im, padval=None):
		if padval==None:
			if im.ndim==2:
				padval = self.flow_pad_value
			else:
				padval = self.RGB_pad_value
		pad_im = self.padarray(im, padval)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.ANTIALIAS)
		return np.array(res_im)

class LabelResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im, BackGroundLabel)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.NEAREST)
		return np.array(res_im)
	
	def upsample(self, im, ImgSize):
		im = Image.fromarray(im)
		if self.nopadding:
			res_im = im.resize(ImgSize, Image.NEAREST)
			return np.array(res_im)

		box_size = self.box_size
		box_size = (max(ImgSize[0], ImgSize[1]), max(ImgSize[0], ImgSize[1])) if box_size==None else box_size
		res_im = im.resize(box_size, Image.NEAREST)
		res_im = np.array(res_im)
		res_im = res_im[0:ImgSize[0], 0:ImgSize[1]]
		return res_im



def LoadLabel(filename, resizer=None):
	im = np.array(Image.open(filename))
	Dtype = im.dtype
	
	if resizer!=None:
		im = resizer.resize(im)
	
	im = im.reshape(im.shape[0],im.shape[1],1)
	im = np.array(im, Dtype)
	#print np.amin(im),np.amax(im), im.shape
	
	im = im.transpose((2,0,1))
	return im

def LoadImage(filename, flow_x=None, flow_y=None, resizer=None):
	im = np.array(Image.open(filename))
	Dtype = im.dtype
	im = im[:,:,::-1]	# reverse channels of image data
	
	if (flow_x!=None):
		flow_im = np.array(Image.open(flow_x))
		flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
		im = np.concatenate((im, flow_im), axis=2)
		
	if (flow_y!=None):
		flow_im = np.array(Image.open(flow_y))
		flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
		im = np.concatenate((im, flow_im), axis=2)
	
	if resizer!=None:
		im_res = resizer.resize(im[:,:,:3])
		for i in range(3, im.shape[2]):
			flow_im_res = resizer.resize(im[:,:,i])
			flow_im_res = np.reshape(flow_im_res, (flow_im_res.shape[0], flow_im_res.shape[1], 1))
			im_res = np.concatenate((im_res, flow_im_res), axis=2)
		im = im_res
		
	im = np.array(im,Dtype)
	im = im.transpose((2,0,1))
	# shape channels*width*height, e.g. 3*640*420
	return im

def createLMDBLabel(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, keys=None, args=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	resizer = None if not resize else LabelResizer(args.LabelSize, args.BoxSize, args.nopadding)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			in_ = inputs_Train[key]
			im = LoadLabel(in_, resizer)
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
			print in_idx, 'label', im.shape
	in_db.close()



def createLMDBImage(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, keys=None, args=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	RGB_sum = np.zeros(3 + (flow_x!=None) + (flow_y!=None))
	resizer = None if not resize else ImageResizer(args.RSize, args.BoxSize, args.nopadding, args.RGB_pad_values, args.flow_pad_value)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			print in_idx
			in_ = inputs_Train[key]
			im = LoadImage(in_, flow_x[key] if flow_x!=None and (key in flow_x) else None, flow_y[key] if flow_y!=None and (key in flow_y) else None, resizer)
			RGB_sum = RGB_sum + np.mean(im, axis=(1,2))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
			print in_idx, im.shape, RGB_sum/(in_idx+1)
	in_db.close()
	f = open(os.path.join(dir, 'RGB_mean'), 'w')
	RGB_mean = RGB_sum/len(keys)
	for i in range(RGB_mean.size):
		f.write('mean_value: ' + str(RGB_mean[i]) + '\n')
	for i in range(RGB_mean.size):
		f.write(str(RGB_mean[i]) + ', ')
	f.close()

