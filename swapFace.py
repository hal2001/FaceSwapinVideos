import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import pdb
import cv2
from scipy.spatial import Delaunay
from getFrame import getFrame
from FaceDetection import FaceDetection
from ConvexHull import ConvexHull
from warping import warping
from morph_tri import morph_tri
from Blend import Blend
from videoGenerate import videoGenerate
from emotionMatching import emotionMatching


def swapFace(dict, img_input1, img_input2):
    output1 = []
    for frame in range(0, len(img_input1)):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_input1[frame])
        ax2.imshow(img_input2[dict[frame]])
        f.savefig("sep %s.jpg" % frame)

        w = img_input1[frame].shape[1]
        h = img_input1[frame].shape[0]
        w2 = img_input2[dict[frame]].shape[1]
        h2 = img_input2[dict[frame]].shape[0]
        if h>h2:
            hu=h2
            wu=w2
            img_input1[frame]=cv2.resize(img_input1[frame],(w2,h2), interpolation=cv2.INTER_CUBIC)
        else:
            hu=h
            wu=w
            img_input2[dict[frame]]=cv2.resize(img_input2[dict[frame]],(w,h), interpolation=cv2.INTER_CUBIC)

        if frame==0:
            bbox1, feature1= FaceDetection(img_input1[frame])
            bbox2, feature2= FaceDetection(img_input2[dict[frame]])
            p0=feature1
            p02=feature2

        elif frame % 15==0 or (feature1.shape)[0]<50:
            bbox1, feature1= FaceDetection(img_input1[frame])
            bbox2, feature2= FaceDetection(img_input2[dict[frame]])
            p0=feature1
            p02=feature2

        else:
            old_gray = cv2.cvtColor(img_input1[frame-1], cv2.COLOR_BGR2GRAY)
            new_gray = cv2.cvtColor(img_input1[frame], cv2.COLOR_BGR2GRAY)
            old_gray2 = cv2.cvtColor(img_input2[dict[frame-1]], cv2.COLOR_BGR2GRAY)
            new_gray2 = cv2.cvtColor(img_input2[dict[frame]], cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p0=p0.astype('float32')
            p02=p02.astype('float32')
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None)#, **lk_params)
            p12, st, err = cv2.calcOpticalFlowPyrLK(old_gray2, new_gray2, p02, None)#, **lk_params)

            # Select good points
            #pdb.set_trace()
            good_new = p1[(st == 1)[:,0],:]
            good_new2 = p12[(st == 1)[:,0],:]

            p0=good_new
            p02=good_new2
            feature1=p1
            feature2=p12
        '''
        plt.figure()
        plt.imshow(img_input1[frame])
        x = feature1[:, 0]
        y = feature1[:, 1]
        plt.plot(x, y)
    
        plt.figure()
        plt.imshow(img_input2[dict[frame]])
        x2 = feature2[:, 0]
        y2 = feature2[:, 1]
        plt.plot(x2, y2)
        plt.show()
        '''
        convH1 = ConvexHull(img_input1[frame], feature1)
        convH2 = ConvexHull(img_input2[dict[frame]], feature2)
        convHfill1 = np.zeros([hu, wu])
        convHfill2 = np.zeros([hu, wu])
        mask1 = cv2.fillConvexPoly(convHfill1, convH1, 1)
        mask2 = cv2.fillConvexPoly(convHfill2, convH2, 1)

        '''
        imagecopy = img_input1[frame].copy()
        mask1=mask1.astype('uint8')
        imagecopy[mask1==1] = [255, 0, 255]
        plt.imshow(imagecopy)
        plt.savefig("mask1.jpg")
    
        imagecopy2 = img_input2[dict[frame]].copy()
        mask2=mask2.astype('uint8')
        imagecopy2[mask2==1] = [255, 0, 255]
        plt.imshow(imagecopy2)
        plt.savefig("mask2.jpg")
        plt.show()
        '''
        f1=np.zeros(np.shape(feature1))
        f2=np.zeros(np.shape(feature2))
        f1[:,0]=feature1[:,1]
        f1[:,1]=feature1[:,0]
        f2[:,0]=feature2[:,1]
        f2[:,1]=feature2[:,0]
        f1 = np.append(f1, np.array([[0, 0], [0, wu - 1], [hu - 1, 0], [hu - 1, wu - 1]]), axis = 0)
        f2 = np.append(f2, np.array([[0, 0], [0, wu - 1], [hu - 1, 0], [hu - 1, wu - 1]]), axis = 0)

        print('Morphing!')
        #img_output1 = warping(img_input1[frame],img_input2[frame],f1, f2, np.array([0]), np.array([1]))
        img_output1 = morph_tri(img_input1[frame], img_input2[dict[frame]], f1, f2, np.array([0]), np.array([1]))

        img_out1 = img_input1[frame].copy()
        img_out1[mask1 == 1] = img_output1[mask1 == 1]
        result=Blend(img_output1, img_out1, mask1)

        plt.figure()
        plt.imshow(result)
        plt.savefig("v1"+"%s.jpg"%frame)
        #plt.show()
        plt.figure()
        plt.imshow(img_out1)
        plt.savefig("before %s.jpg" %frame)
        #plt.show()

        output1.append(result)
    return output1

if __name__ == '__main__':
    #img_input1, img_input2 = getFrame('Easy')
    img_input1=np.load('img_input1.npy')
    img_input2=np.load('img_input2.npy')
    #dict = emotionMatching(img_input1, img_input2)
    dict2 = emotionMatching(img_input2, img_input1)

    #output1=swapFace(dict, img_input1, img_input2)
    output2=swapFace(dict2, img_input2, img_input1)

    #videoGenerate(output1, 1)
    videoGenerate(output2, 2)
