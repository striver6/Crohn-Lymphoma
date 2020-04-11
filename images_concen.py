import cv2
import numpy as np
from keras_preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt





imgs = np.load('LYM_sfa\\results2\\imgs_mask_train_pred.npy')
img_labels = np.load('LYM_sfa\\npydata\\imgs_mask_train.npy')
imgs_mask_test = imgs


def result_map_to_img(img, res_map):
    res_map = np.squeeze(res_map)
    liver = (res_map == 0)
    ascites = (res_map == 255)
    # liver = (res_map == 128)
    # ascites = (res_map > 220)

    img[liver, 0] = 0
    img[liver, 1] = 255
    img[liver, 2] = 0

    img[ascites, 0] = 0          #255
    img[ascites, 1] = 0
    img[ascites, 2] = 255         #255

    return img

for i in range(imgs.shape[0]):

    img = imgs[i].astype(int)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img_to_array(img)
    # print(np.mean(img),"--",np.max(img),"--",np.min(img))
    # print(img[img>0])
    # img = np.expand_dims(img,axis=2)
    # img = np.concatenate((img,img,img),axis=-1)
    print(img.shape)

    img_label = img_labels[i]
    # img_label = array_to_img(img_label)
    # img_label = img_label.convert("RGB")
    # img_label = img_to_array(img_label)
    # img_label = np.expand_dims(img_label,axis=2)
    # img_label = np.concatenate((img_label,img_label,img_label),axis=-1)
    print(img_label.shape)
    # merge = img+img_label
    cv2.imwrite("change/{}.png".format(str(i)),result_map_to_img(img,img_label))


    # print('--------')
    #
    # height = img_mask_test.shape[0]
    # weight = img_mask_test.shape[1]
    # # channels = imgs_mask_test.shape[3]
    # print(img_mask_test.shape)
    # for row in range(height):            #遍历高
    #     for col in range(weight):         #遍历宽
    #         img_mask_test = img_mask_test.astype(int)
    #         # print(np.max(img),"--",np.min(img))
    #         if(img_mask_test[row, col,0]==255):
    #             img_mask_test[row, col, 0] = 255
    #             img_mask_test[row, col, 1] = 0
    #             img_mask_test[row, col, 2] = 0
    # plt.imshow(img_mask_test)
    # plt.margins(0, 0)
    # plt.savefig("change/{}.png".format(str(i)))
    # plt.show()






