import cv2
import numpy as np
from keras_preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np
import os
import glob
datasets_path = ''
class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path = "merge",
             label_path = "train\label", test_path = "test",
             npy_path = "merge1", img_type = "png"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = datasets_path+data_path+"\\"
        self.label_path = datasets_path+label_path+"\\"
        self.img_type = img_type
        self.test_path = datasets_path+test_path+"\\"
        self.npy_path = datasets_path+npy_path+"\\"

    def create_train_data(self):
        print('-'*30)
        print('Creating training images...')
        print(self.data_path)
        imgs = glob.glob(self.data_path+"*."+self.img_type)[:2000]
        print(imgs)
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
        # imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols,1), dtype=np.uint8)
        for i in range(len(imgs)):
            img = load_img(self.data_path + "{}.png".format(i),grayscale = False)
            # label = load_img(self.label_path + "{}.png".format(i),grayscale = True)
            img = img_to_array(img)
            # label = img_to_array(label)
            imgdatas[i] = img
            # label[label > 0] = 1
            # label[label == 0] = 0
            # imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
        print('loading done')
        np.save(self.npy_path + 'imgs_train.npy', imgdatas)
        # np.save(self.npy_path + 'imgs_train_mask.npy', imglabels)
        print('Saving to imgs_mask_train.npy files done.')


# data = dataProcess(512,512)
# data.create_train_data()

#
# import os
# for root,dirs,files in os.walk("train\image"):
#     # for dir in dirs:
#     for file in files:
#         # os.walk(os.path.join(root, dir)):
#         if(file.endswith("png")): `q!a
#             print(file)
#             # len1.append(len(files[2]))



imgs = np.load("train\\npydata\\"+"imgs_train.npy")
# imgs = np.load("merge1\\"+"imgs_train.npy")
img_labels = np.load("results1\\imgs_mask_train_pred.npy")
# img_labels = np.load('train\\npydata\\imgs_mask_train.npy')

# imgs = np.load("merge2\\"+"imgs_train.npy")
# img_labels = np.load("train_processed1\\"+"imgs_train_mask_processed.npy")
# imgs = np.load('merge\\imgs_train.npy')
# img_labels = np.load('merge\\imgs_train_mask.npy')
# imgs = np.load('train\\npydata\\imgs_train.npy')
# img_labels = np.load('train\\npydata\\imgs_mask_train.npy')
# imgs_mask_test = imgs


def result_map_to_img(img, res_map):
    res_map = np.squeeze(res_map)
    liver = (res_map == 0)
    ascites = (res_map == 1)
    # liver = (res_map == 128)
    # ascites = (res_map > 220)

    # img[liver, 0] = 0
    # img[liver, 1] = 255
    # img[liver, 2] = 0

    img[ascites, 0] = 0        #255
    img[ascites, 1] = 0
    img[ascites, 2] = 255         #255

    return img

for i in range(imgs.shape[0]):

    img = imgs[i].astype(int)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img_to_array(img)
    print(img.shape)
    # print(img[img>0])
    # img = np.expand_dims(img,axis=2)
    # img = np.concatenate((img,img,img),axis=-1)

    img_label = img_labels[i]
    print(img_label.shape)
    # print(img_label)
    cv2.imwrite("merge1/{}.png".format(str(i)),result_map_to_img(img,img_label))
    # cv2.imwrite("merge/{}.png".format(str(i)),img)
    # cv2.imwrite("merge/{}.png".format(str(i)),img_label)







