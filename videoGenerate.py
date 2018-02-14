import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pdb
from PIL import Image
import matplotlib.pyplot as plt


def videoGenerate(input, name):
	plt.rcParams['animation.ffmpeg_path'] = './ffmpeg'

	Res = np.ones([len(input), input[0].shape[0], input[0].shape[1], 3], dtype = np.uint8)
	for j in range(0, len(input)):
		Res[j, :, :, :] = input[j]

	ims = []
	fig = plt.figure()
	plt.axis('off')
	for i in range(len(input)):
		im = plt.imshow(Res[i], animated = True)
		ims.append([im])

	#ani = animation.ArtistAnimation(fig, ims, blit = True)
	ani = animation.ArtistAnimation(fig, ims, interval = 40, blit = True)
	myWriter = animation.FFMpegWriter()
	ani.save('%s.avi' % name, writer = myWriter)
	plt.show()
