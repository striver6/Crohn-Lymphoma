
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout,concatenate,BatchNormalization,AveragePooling2D,LeakyReLU,MaxPool2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from data import *
import keras.backend.tensorflow_backend as KTF


from numba import jit

import warnings
warnings.filterwarnings('ignore')

import os


def dice_coef(y_true, y_pred, smooth=1, weight=1):
    """
    加权后的dice coefficient
    """
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    目标函数
    """
    return 1 - dice_coef(y_true, y_pred)

def f1_score(y_true, y_pred, smooth=1):
    """
    f1 score，用于训练过程中选择模型
    """
    y_true = y_true[:,:,:,-1]
    y_pred = y_pred[:,:,:,-1]
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    f1_score = (2*c1+smooth)/(c2+c3+smooth)
    return f1_score

class myUnet(object):

    def __init__(self, img_rows = 512, img_cols = 512):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test


    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,1))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(1, 1), kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', dilation_rate=(1, 1),kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)


        model = Model(input = inputs, output = conv10)
        print(model.summary())
        model.load_weights("LYM_vfa/aug_data/unet-2019-12-28-21-38-12-0.47.hdf5")
        # model.compile(optimizer = Adam (lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = dice_coef_loss, metrics = [f1_score])
        model.compile(optimizer = Adam (lr=1e-5), loss = dice_coef_loss, metrics = [f1_score])

        return model

    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()

        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('LYM_vfa/aug_data/unet-{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M'))+'-{epoch:02d}-{val_f1_score:.2f}.hdf5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        print(imgs_train.shape,imgs_mask_train.shape)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-5)
        hist = model.fit(imgs_train, imgs_mask_train, batch_size=1, nb_epoch=300, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint,reduce_lr])
        with open('LYM_vfa/aug_data/log_{}.txt'.format(datetime.now().strftime('%Y-%m-%d-%H-%M')), 'w') as f:
            f.write(str(hist.history)+'\n')
        print('predict test data')
        print(imgs_test.shape)
        imgs_mask_test = model.predict(imgs_test, batch_size=32, verbose=1)
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('results1\\imgs_mask_test.npy', imgs_mask_test)
        np.save('LYM_vfa\\aug_data\\results1\\imgs_mask_test_pred.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        # imgs = np.load('results1\\imgs_mask_test.npy')
        imgs = np.load('LYM_vfa\\aug_data\\results1\\imgs_mask_test_pred.npy')
        # imgs = np.load('LYM_vfa\\results1\\imgs_mask_train.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            # img.save("results1\\%d.jpg"%(i))
            img.save("LYM_vfa\\aug_data\\results1\\%d.jpg"%(i))

from datetime import datetime
import tensorflow as tf
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"



    start = datetime.now()
    myunet =myUnet()
    myunet.train()

    stop = datetime.now()
    # print(stop - start)
    print('Training time cost: %0.2f(min).' % ((stop - start) / 60))







