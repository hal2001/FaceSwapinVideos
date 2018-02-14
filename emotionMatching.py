import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import pdb
from getFrame import getFrame
from FaceDetection import FaceDetection
import numpy.matlib


def emotionMatching(img_input1, img_input2):
    dict = {}
    all_feature2= np.zeros([1,62*2])
    for i in range(0, len(img_input2)):
        print("frame %s"%i)
        bbox2, feature2 = FaceDetection(img_input2[i])
        h0, w0 = np.shape(feature2)
        center2 = np.mean(feature2, axis = 0)
        #pdb.set_trace()
        flat_f2=np.transpose(np.reshape(feature2-center2, h0*2,1))
        flat_f2=np.matrix(flat_f2)
        all_feature2=np.append(all_feature2,flat_f2, axis=0)
    h,w=np.shape(all_feature2)
    for frame in range(0, len(img_input1)):
        print('finding match for %s frame !!!!' %frame)
        bbox1, feature1 = FaceDetection(img_input1[frame])
        center1 = np.mean(feature1, axis=0)
        h0,w0=np.shape(feature1)
        flat_f1 = np.transpose(np.reshape(feature1 - center1, h0 * 2, 1))
        flat_f1 = np.matrix(flat_f1)
        all_feature1=np.matlib.repmat(flat_f1, h, 1)
        diff=np.asarray(all_feature1-all_feature2)
        diff=np.sum(diff**2, axis=1)
        dict[frame]=np.argmin(diff)


        #diff = np.sum((feature1 - center1 - (feature2 - center2)) ** 2)
        #print("for frame %s in 1, we are checking 2's frame %s diff is %s" %(frame, i, diff))
            #if diff < minDis:
            #   minDis = diff
            #   minN = i
        #if minDis < 10000:
        #dict[frame] = minN

    return dict


if __name__ == '__main__':
    img_input1, img_input2 = getFrame('Easy')
    dict = emotionMatching(img_input1, img_input2)
    for i in range(0, len(img_input1)):
        #if i in dict:
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_input1[i])
        ax2.imshow(img_input2[dict[i]])
        f.show()
