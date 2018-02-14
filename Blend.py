#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/15/17 10:12 PM 

@author: Hantian Liu
"""
import numpy as np
from PIL import Image
#from maskImage import maskImage
from getIndexes import getIndexes
import scipy.misc
import scipy.io as sio
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg
import matplotlib.pyplot as plt
from seamlessCloningPoisson import seamlessCloningPoisson

def Blend(img1, img2, mask):
	'''
	# resize the image
	[h,w,z]=np.shape(simg)
	h=int(0.4*h)
	w=int(0.4*w)
	sim = scipy.misc.imresize(simg, [h, w])
	'''
	offsetX=0
	offsetY=0


	result=seamlessCloningPoisson(img1, img2, mask, offsetX, offsetY)

	return result