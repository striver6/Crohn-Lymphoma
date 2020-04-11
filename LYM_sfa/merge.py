import cv2
import numpy as np
from keras_preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img




def create_npy():
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    imgs = glob.glob("merge//*.png" )
    imgdatas = np.ndarray((len(imgs), 512, 512, 3), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = load_img("merge"+ "\\" + midname, grayscale=False)
        img = img_to_array(img)
        imgdatas[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print('loading done')
    np.save('merge/imgs_merge.npy', imgdatas)
    print('Saving to imgs_merge.npy files done.')

# create_npy()




def result_map_to_img(img, res_map):
    res_map = np.squeeze(res_map)
    liver = (res_map == 0)
    ascites = (res_map == 1)
    # liver = (res_map == 128)
    # ascites = (res_map > 220)

    # img[liver, 0] = 0
    # img[liver, 1] = 255
    # img[liver, 2] = 0

    img[ascites, 0] = 0          #255
    img[ascites, 1] = 0
    img[ascites, 2] = 255         #255

    return img



imgs = np.load('train_aug\\npydata\\imgs_train.npy')
# imgs = np.load('merge\\imgs_merge.npy')
img_labels = np.load('results6\\npy\\imgs_mask_train_pred.npy')
# img_labels = np.load('train\\npydata\\imgs_mask_train.npy')
imgs_mask_test = imgs

import skimage.io
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes

for i in range(imgs.shape[0]):

    img = imgs[i].astype(int)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img_to_array(img)
    print(img.shape)

    img_label = img_labels[i]
    print(img_label.shape)

    image = skimage.morphology.remove_small_objects(label(img_label), min_size=500, connectivity=1, in_place=False)
    image = img_to_array(image)
    res_map = np.squeeze(image)
    liver = (res_map == 0)
    ascites = (res_map > 0)
    image[liver] = 0
    image[ascites] = 1
    img_label = image


    cv2.imwrite("merge5/{}.png".format(str(i)),result_map_to_img(img,img_label))
    # cv2.imwrite("merge5/{}_original.png".format(str(i)),img)







