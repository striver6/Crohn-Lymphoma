from unet import *
from data import *
import matplotlib.pylab as plt


imgs = np.load('LYM\\npydata\\imgs_train.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    # print(img)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img.point(lambda i: i*2000)
    # print(img)
    print('+'*100)
    print(img.info)
    print(img.size)
    print(img.mode)
    plt.imshow(img)
    plt.show()
    img.save("LYM\\results2\\train_%d_original.jpg" % (i))


imgs = np.load('LYM\\npydata\\imgs_test.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    # print(img)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img.point(lambda i: i*2000)
    # print(img)
    print('+'*100)
    print(img.info)
    print(img.size)
    print(img.mode)
    plt.imshow(img)
    plt.show()
    img.save("LYM\\results2\\test_%d_original.jpg" % (i))