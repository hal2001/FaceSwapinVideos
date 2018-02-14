'''
  File name: morph_tri.py
  Author: Jinglei Yu
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
import scipy.io
import os
from PIL import Image
from numpy.linalg import inv
import pdb
import matplotlib.pyplot as plt


def warping(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    # Tips: use Delaunay() function to get Delaunay triangulation;
    # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
    IMAGE_HEIGHT = im1.shape[0]
    IMAGE_WIDTH = im1.shape[1]

    frame = 1
    morphed_im = np.empty([frame, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.uint8)

    Y = np.arange(0, IMAGE_HEIGHT, 1)
    X = np.arange(0, IMAGE_WIDTH, 1)
    xx, yy = np.meshgrid(X, Y)
    xx_f = xx.flatten()
    yy_f = yy.flatten()

    # find the inter_pts in the middle
    inter_pts = np.empty([frame, im1_pts.shape[0], im1_pts.shape[1]])
    for frame in range(0, 1):
        inter_pts[frame] = im1_pts * (1 - warp_frac[frame]) + im2_pts * warp_frac[frame]
        tri = Delaunay(inter_pts[frame])

        # get the positions of triangulation index for each control point
        # shape:[numTri,3,2]
        numTri = inter_pts[frame][tri.simplices].shape[0]
        matrix_all_s = np.empty([numTri, 3, 3])
        matrix_all_t = np.empty([numTri, 3, 3])
        matrix_all_i = np.empty([numTri, 3, 3])
        matrix_inv_i = np.empty([numTri, 3, 3])

        for num_tri in range(0, numTri):
            # culculate the matrix of source image
            matrix_up_s = np.transpose(im1_pts[tri.simplices][num_tri])
            matrix_all_s[num_tri] = np.append(matrix_up_s, [[1, 1, 1]], axis=0)

            # culculate the matrix of target image
            matrix_up_t = np.transpose(im2_pts[tri.simplices][num_tri])
            matrix_all_t[num_tri] = np.append(matrix_up_t, [[1, 1, 1]], axis=0)

            # culculate the matrix of inter image
            matrix_up_i = np.transpose(inter_pts[frame][tri.simplices][num_tri])
            matrix_all_i[num_tri] = np.append(matrix_up_i, [[1, 1, 1]], axis=0)
            matrix_inv_i[num_tri] = inv(matrix_all_i[num_tri])

        # check which traingle contains the query point with tri.find_simplex
        # compute the alpha beta gama for all images of each pixel
        Pos_s_homo = np.empty([IMAGE_HEIGHT * IMAGE_WIDTH, 2])
        Pos_t_homo = np.empty([IMAGE_HEIGHT * IMAGE_WIDTH, 2])
        num_p = 0

        for y in range(0, IMAGE_HEIGHT):
            for x in range(0, IMAGE_WIDTH):
                    # compute the alpha beta gama for all images of each pixel
                    Bary_cor = np.dot(matrix_inv_i[tri.find_simplex(np.array([x, y]))], np.array([[x], [y], [1]]))

                    # compute the corresponding pixel position in the source image
                    Pos_s = np.dot(matrix_all_s[tri.find_simplex(np.array([x, y]))], Bary_cor)
                    Pos_s_h = Pos_s / Pos_s[2]

                    Pos_t = np.dot(matrix_all_t[tri.find_simplex(np.array([x, y]))], Bary_cor)
                    Pos_t_h = Pos_t / Pos_t[2]
                    # Pos_s_homo[num_p] shape:[1,2]
                    Pos_s_homo[num_p] = np.transpose(Pos_s_h[0:2])
                    Pos_t_homo[num_p] = np.transpose(Pos_t_h[0:2])
                    # check whether positions are out of boundary
                    if Pos_s_homo[num_p][0] < 0:
                        Pos_s_homo[num_p][0] = 0
                    if Pos_s_homo[num_p][1] < 0:
                        Pos_s_homo[num_p][1] = 0
                    if Pos_s_homo[num_p][0] > IMAGE_WIDTH - 1:
                        Pos_s_homo[num_p][0] = IMAGE_WIDTH - 1
                    if Pos_s_homo[num_p][1] > IMAGE_HEIGHT - 1:
                        Pos_s_homo[num_p][1] = IMAGE_WIDTH - 1

                    if Pos_t_homo[num_p][0] < 0:
                        Pos_t_homo[num_p][0] = 0
                    if Pos_t_homo[num_p][1] < 0:
                        Pos_t_homo[num_p][1] = 0
                    if Pos_t_homo[num_p][0] > IMAGE_WIDTH - 1:
                        Pos_t_homo[num_p][0] = IMAGE_WIDTH - 1
                    if Pos_t_homo[num_p][1] > IMAGE_HEIGHT - 1:
                        Pos_t_homo[num_p][1] = IMAGE_WIDTH - 1
                    num_p = num_p + 1

        # find the intermediate points shape
        morphed_im_s = np.empty([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        morphed_im_t = np.empty([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        # inter_index shape:[1,IMAGE_HEIGHT*IMAGE_WIDTH]

        interm_pos_x = Pos_s_homo[:, 0]
        interm_pos_y = Pos_s_homo[:, 1]
        interm_pos_x = np.round(interm_pos_x).astype(int)
        interm_pos_y = np.round(interm_pos_y).astype(int)
        inter_index = interm_pos_y * IMAGE_WIDTH + interm_pos_x

        interm_pos_t_x = Pos_t_homo[:, 0]
        interm_pos_t_y = Pos_t_homo[:, 1]
        interm_pos_t_x = np.round(interm_pos_t_x).astype(int)
        interm_pos_t_y = np.round(interm_pos_t_y).astype(int)
        inter_t_index = interm_pos_t_y * IMAGE_WIDTH + interm_pos_t_x

        # reshape mag to shape: [IMAGE_HEIGHT,IMAGE_WIDTH]
        r_s = np.reshape(im1[:, :, 0].flatten()[inter_index], (IMAGE_HEIGHT, IMAGE_WIDTH))
        g_s = np.reshape(im1[:, :, 1].flatten()[inter_index], (IMAGE_HEIGHT, IMAGE_WIDTH))
        b_s = np.reshape(im1[:, :, 2].flatten()[inter_index], (IMAGE_HEIGHT, IMAGE_WIDTH))

        morphed_im_s[:, :, 0] = r_s.astype(np.uint8)
        morphed_im_s[:, :, 1] = g_s.astype(np.uint8)
        morphed_im_s[:, :, 2] = b_s.astype(np.uint8)

        r_t = np.reshape(im2[:, :, 0].flatten()[inter_t_index], (IMAGE_HEIGHT, IMAGE_WIDTH))
        g_t = np.reshape(im2[:, :, 1].flatten()[inter_t_index], (IMAGE_HEIGHT, IMAGE_WIDTH))
        b_t = np.reshape(im2[:, :, 2].flatten()[inter_t_index], (IMAGE_HEIGHT, IMAGE_WIDTH))

        morphed_im_t[:, :, 0] = r_t.astype(np.uint8)
        morphed_im_t[:, :, 1] = g_t.astype(np.uint8)
        morphed_im_t[:, :, 2] = b_t.astype(np.uint8)

        morphed_im[frame] = morphed_im_s * (1 - dissolve_frac[frame]) + morphed_im_t * dissolve_frac[frame]
        print(frame)
        plt.imshow(morphed_im[frame])
        plt.savefig("%s.jpg" % frame)
        plt.show()

    return morphed_im


if __name__ == '__main__':

    # read images one by one
    im1 = np.array(Image.open('im1.jpg').convert('RGB'))
    im2 = np.array(Image.open('im2.jpg').convert('RGB'))

    im1_pts, im2_pts = click_correspondences(im1, im2)
    warp_frac = np.arange(0.0, 1.0, 1 / 60)
    dissolve_frac = np.arange(0.0, 1.0, 1 / 60)
    morphed_im = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)
    # scipy.io.savemat("morphed_im.mat",mdict={'morphed_im': morphed_im[30]})
    # pdb.set_trace()
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6, figsize=(12, 12))

    # plot original image
    ax0.imshow(morphed_im[0])
    ax0.axis("off")
    ax0.set_title('0')

    # plot edge detection result
    ax1.imshow(morphed_im[15])
    ax1.axis("off")
    ax1.set_title('15')

    # plot original image
    ax2.imshow(morphed_im[25])
    ax2.axis("off")
    ax2.set_title('25')

    ax3.imshow(morphed_im[35])
    ax3.axis("off")
    ax3.set_title('35')

    # plot edge detection result
    ax4.imshow(morphed_im[45])
    ax4.axis("off")
    ax4.set_title('45')

    # plot original image
    ax5.imshow(morphed_im[59])
    ax5.axis("off")
    ax5.set_title('59')

    # imgplot = plt.imshow(morphed_im[59])
    plt.show()
