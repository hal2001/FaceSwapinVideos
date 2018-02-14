import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import pdb


def getFrame(folder):
    img_input1 = []
    img_input2 = []
    numV = 0
    length = []
    lastone = np.array([0])

    # find the max length of videos
    for filename in os.listdir(folder):
        im_path = os.path.join(folder, filename)
        reader = imageio.get_reader(im_path)
        length.append(len(reader))
    maxi = max(length)
    print("maximum length")
    print(maxi)

    # get img_input
    for filename in os.listdir(folder):
        # read in video
        im_path = os.path.join(folder, filename)
        reader = imageio.get_reader(im_path)

        # if it is the first video
        if numV == 0:
            for i, im in enumerate(reader):
                img_input1.append(im)
                lastone = im
            lennow = len(reader)
            print("first video length")
            print(lennow)
            while lennow < maxi:
                img_input1.append(lastone)
                lennow = lennow + 1
            numV = numV + 1

        # next videos
        else:
            for i, im in enumerate(reader):
                img_input2.append(im)
                lastone = im
            lennow = len(reader)
            print("second video length")
            print(lennow)
            while lennow < maxi:
                img_input2.append(lastone)
                lennow = lennow + 1

    print("import video done!")
    return img_input1, img_input2


if __name__ == '__main__':
    img_input1, img_input2 = getFrame('Easy')
    plt.imshow(img_input1[0])
    plt.show()
    plt.imshow(img_input2[0])
    plt.show()
