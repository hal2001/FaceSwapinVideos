#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/12/17 9:26 PM 

@author: Hantian Liu
"""

'''
input: Gray Image (h*w)
output: bounding box (2*2) [x_upper_left, y_upper_left; x_lower_right, y_lower_right]
'''

import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from ConvexHull import ConvexHull
import matplotlib.pyplot as plt
from MaskConvexHull import MaskConvexHull
import cv2
from morph_tri import morph_tri
from Blend import Blend

def FaceDetection(img):
	face_locations = face_recognition.face_locations(img)
	num = len(face_locations)
	print("found {} face(s) in this photograph!".format(num))
	# bbox=np.zeros([num,4,2])
	bbox = []
	all_feature = []
	for face_location in face_locations:  # iterate once only

		# Print the location of each face in this image
		top, right, bottom, left = face_location
		#print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
		#																							right))

		bbox.append(np.array([[top, left], [top, right], [bottom, right], [bottom, left]]))

		# You can access the actual face itself like this:
		#face_image = image[top:bottom, left:right]
		#pil_image = Image.fromarray(face_image)
		#pil_image.show()

	face_landmarks_list = face_recognition.face_landmarks(img)
	for face_landmarks in face_landmarks_list: #iterate once only

		# Print the location of each facial feature in this image
		facial_features = [
			'chin',
			'left_eyebrow',
			'right_eyebrow',
			'nose_bridge',
			'nose_tip',
			'left_eye',
			'right_eye',
			'top_lip',
			'bottom_lip'
		]
		feature = np.zeros([1,2])
		for facial_feature in facial_features:
			#print("The {} in this face has the following points: {}".format(facial_feature,
			#																face_landmarks[facial_feature]))
			# Let's trace out each facial feature in the image with a line!
			#pil_image = Image.fromarray(img)
			#d = ImageDraw.Draw(pil_image)
			this_feature=np.asarray(face_landmarks[facial_feature])
			if facial_feature=='chin':
				this_feature=this_feature[5:12]
			#elif facial_feature=='left_eyebrow':
			#	this_feature=this_feature[1:4]
			#elif facial_feature=='right_eyebrow':
			#	this_feature=this_feature[0:3]
			feature=np.append(feature,this_feature,axis=0)
			#for facial_feature in facial_features:
			#	d.line(face_landmarks[facial_feature], width = 5)

	#pil_image.show()
	#print('we have')
	feature=feature[1:,:]
	#print(feature)
	#print(bbox)
	return bbox, feature

if __name__ == '__main__':
	image = face_recognition.load_image_file("a.jpg")
	image2 = face_recognition.load_image_file("test2.jpg")
	bbox, feature = FaceDetection(image)
	bbox2, feature2 = FaceDetection(image2)
	'''
	plt.figure()
	plt.imshow(image)
	x=feature[:,0]
	y=feature[:,1]
	plt.plot(x,y)

	plt.figure()
	plt.imshow(image2)
	x2 = feature2[:, 0]
	y2 = feature2[:, 1]
	plt.plot(x2, y2)

	plt.show()
	'''
	#hullpts=ConvexHull(image, feature)

	hu, wu,z=np.shape(image)
	convH1 = ConvexHull(image, feature)
	convH2 = ConvexHull(image2, feature2)
	convHfill1 = np.zeros([hu, wu])
	convHfill2 = np.zeros([hu, wu])
	mask1 = cv2.fillConvexPoly(convHfill1, convH1, 1)
	mask2 = cv2.fillConvexPoly(convHfill2, convH2, 1)
	f1 = np.zeros(np.shape(feature))
	f2 = np.zeros(np.shape(feature2))
	f1[:, 0] = feature[:, 1]
	f1[:, 1] = feature[:, 0]
	f2[:, 0] = feature2[:, 1]
	f2[:, 1] = feature2[:, 0]
	f1 = np.append(f1, np.array([[0, 0], [0, wu - 1], [hu - 1, 0], [hu - 1, wu - 1]]), axis = 0)
	f2 = np.append(f2, np.array([[0, 0], [0, wu - 1], [hu - 1, 0], [hu - 1, wu - 1]]), axis = 0)

	print('Warping!')
	# img_output1 = warping(img_input1[frame],img_input2[frame],f1, f2, np.array([0]), np.array([1]))
	img_output1 = morph_tri(image, image2, f1, f2, np.array([0]), np.array([1]))
	img_output2 = morph_tri(image2, image, f2, f1, np.array([0]), np.array([1]))

	img_out1 = image.copy()
	img_out2 = image2.copy()
	img_out1[mask1 == 1] = img_output1[mask1 == 1]
	result = Blend(img_output1, img_out1, mask1)

	img_out2[mask2 == 1] = img_output2[mask2 == 1]
	result2 = Blend(img_output2, img_out2, mask2)

	plt.imshow(result)
	plt.savefig("ares.jpg")
	plt.imshow(result2)
	plt.savefig("ares2.jpg")

	#mask=MaskConvexHull(image, hullpts)
	#imagecopy=image.copy()
	#mask.astype('uint8')
	#imagecopy[mask]=[255,0,255]
	# plt.imshow(imagecopy)
	# plt.show()

	'''
	plt.imshow(image)
	x = hullpts[:, 0]
	y = hullpts[:, 1]
	plt.plot(x, y)
	plt.show()
	
	bbox2, feature2 = FaceDetection(image2)
	hullpts2 = ConvexHull(image2, feature2)
	print(feature)
	print(feature2)
	'''
