from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
import matplotlib.pylab as plt

# datasets_path = "data"+"\\"
datasets_path = "LYM_sfa"+"\\"

class dataProcess(object):

	# def __init__(self, out_rows, out_cols, data_path = "train\image",
	# 			 label_path = "train\label", test_path = "test",
	# 			 npy_path = "npydata", img_type = "jpg"):
	def __init__(self, out_rows, out_cols, data_path = "train\image",
			 label_path = "train\label", test_path = "test",
			 npy_path = "npydata", img_type = "png"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = datasets_path+data_path+"\\"
		# self.label_path = datasets_path+label_path+"\\"
		self.label_path = datasets_path+label_path
		self.img_type = img_type
		self.test_path = datasets_path+test_path+"\\"
		self.npy_path = datasets_path+npy_path+"\\"

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		print(self.data_path)
		imgs = glob.glob(self.data_path+"*."+self.img_type)[:2000]
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		# imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		# imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("\\")+1:]
			# print(midname)
			img = load_img(self.data_path + "\\" + midname,grayscale = True)
			# label = load_img(self.label_path + "\\" + midname[:-4] + ".png",grayscale = True)
			# label = load_img(self.label_path + "\\" + midname[:-4] + "_mask.jpg",grayscale = True)
			label = load_img(self.label_path + "\\" + midname[:-4] + "_sfa.jpg",grayscale = True)
			img = img_to_array(img)
			label = img_to_array(label)
			# img = img_to_array(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4))
			# label = img_to_array(cv2.resize(label, (512, 512), interpolation=cv2.INTER_LANCZOS4))
			# print(img.shape)
			# print(label.shape)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + 'imgs_train.npy', imgdatas)
		np.save(self.npy_path + 'imgs_mask_train.npy', imglabels)
		print('Saving to imgs_mask_train.npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		print(self.test_path+"*."+self.img_type)
		imgs = glob.glob(self.test_path+"*."+self.img_type)			# ../data_set/train
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		# imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("\\")+1:]
			img = load_img(self.test_path + "\\" + midname,grayscale = True)  		#
			img = img_to_array(img)
			# img = img_to_array(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4))
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + 'imgs_test.npy', imgdatas)			#
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')

		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean
		print("np.max(imgs_train):",np.max(imgs_train))
		print("np.min(imgs_train):",np.min(imgs_train))
		print("np.mean(imgs_train):",np.mean(imgs_train))


		imgs_mask_train /= 255
		# imgs_mask_train[imgs_mask_train > 0.5] = 1
		# imgs_mask_train[imgs_mask_train <= 0.5] = 0
		print("max:",np.max(imgs_mask_train))
		print("min:",np.min(imgs_mask_train))
		print("mean:",np.mean(imgs_mask_train))
		imgs_mask_train[imgs_mask_train > 0] = 1
		imgs_mask_train[imgs_mask_train <= 0] = 0
		print("max:",np.max(imgs_mask_train))
		print("min:",np.min(imgs_mask_train))
		print("mean:",np.mean(imgs_mask_train))

		np.save('LYM_sfa\\train_processed\\imgs_train_processed.npy',imgs_train)
		imgs = np.load('LYM_sfa\\train_processed\\imgs_train_processed.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img = img.convert("RGB")
			img.save("LYM_sfa\\train_processed\\train_%d_processed.jpg" % (i))

		np.save('LYM_sfa\\train_processed\\imgs_train_mask_processed.npy', imgs_mask_train)
		imgs = np.load('LYM_sfa\\train_processed\\imgs_train_mask_processed.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img = img.convert("RGB")
			img.save("LYM_sfa\\train_processed\\train_mask_%d_processed.jpg" % (i))

		return imgs_train,imgs_mask_train




	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean
		return imgs_test

if __name__ == "__main__":

	mydata = dataProcess(512,512)
	mydata.create_train_data()
	mydata.create_test_data()
	mydata.load_train_data()