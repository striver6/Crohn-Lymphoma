import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
I = mpimg.imread('D:/code/Semantic-segmentation/2015/unet/unet-first/LYM_vfa/train/label/14290000_81299956_vfa.jpg')
print(I.shape)
I1 = I/255
# I /= 255
print(np.max(I1))
print(np.mean(I1))
print(np.min(I1))
# plt.imshow(I)
# plt.show()