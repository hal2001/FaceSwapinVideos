#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/12/17 9:29 PM 

@author: Hantian Liu
"""
'''
input: feature points (N*2) [x1, y1; ...; xn, yn]
output: convex hull 
'''
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pdb

def ConvexHull(img, points):
	# points is numpy array of points obtained
	# using dlib.
	points=points.astype('int32')
	hullpts = cv2.convexHull(points)
	#hull = ConvexHull(points)
	# hullIndex is a vector of indices of points
	# that form the convex hull.
	#hullpts=points[hullIndex,:]

	return hullpts[:,0,:]

if __name__ == '__main__':
	points = np.random.rand(30, 2)  # 30 random points in 2-D
	hull = ConvexHull(points)
	plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw = 2)
	plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
	plt.show()