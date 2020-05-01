from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
import matplotlib.pylab as plt

# datasets_path = "data"+"\\"
# datasets_path = "LYM_sfa"+"\\"
# datasets_path = "LYM_sfa"+"\\"
datasets_path = "LYM_sfa"+"\\"
class myAugmentation(object):

	def __init__(self, train_path="train\\image", label_path="agu_data\\label",
			 merge_path="merge",aug_merge_path="aug_merge", aug_train_path="aug_train",
			 aug_label_path="aug_label", img_type="png"):


		self.train_imgs = glob.glob(datasets_path+train_path+"\\"+"*."+img_type)
		self.label_imgs = glob.glob(datasets_path+label_path+"\\"+"*."+img_type)
		self.train_path = datasets_path+train_path+"\\"
		self.label_path = datasets_path+label_path+"\\"
		self.merge_path = datasets_path+merge_path+"\\"
		self.img_type = img_type
		self.aug_merge_path = datasets_path+aug_merge_path+"\\"
		self.aug_train_path = datasets_path+aug_train_path+"\\"
		self.aug_label_path = datasets_path+aug_label_path+"\\"
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		aug_label_path = self.aug_label_path
		print('%d images \n%d labels' % (len(trains), len(labels)))
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print("trains can't match labels")
			return 0

		if not os.path.lexists(path_merge):
			os.mkdir(path_merge)
		if not os.path.lexists(path_aug_merge):
			os.mkdir(path_aug_merge)
		for i in range(len(trains)):
			img_t = load_img(path_train+"\\"+str(i)+"."+imgtype)	# 读入train
			img_l = load_img(path_label+"\\"+str(i)+"."+imgtype)	 # 读入label
			x_t = img_to_array(img_t)								 # 转换成矩阵
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]									 # 把label当做train的第三个通道
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"\\"+str(i)+"."+imgtype)		# 保存合并后的图像
			img = x_t
			img = img.reshape((1,) + img.shape)						 # 改变shape(1, 512, 512, 3)
			savedir = path_aug_merge + "\\" + str(i)				# 存储合并增强后的图像
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))					 # 数据增强

		if not os.path.lexists(aug_label_path):
			os.mkdir(aug_label_path)
		for i in range(len(labels)):
			img_l = load_img(path_label+"\\"+str(i)+"."+imgtype)	 # 读入label
			x_l = img_to_array(img_l)
			img = x_l
			img = img.reshape((1,) + img.shape)						 # 改变shape(1, 512, 512, 3)
			savedir = aug_label_path + "\\" + str(i)				# 存储增强后的图像
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))					 # 数据增强





	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='jpg',
					 imgnum=20):

		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format,shuffle=False,seed=1003):
		    i += 1
		    if i > imgnum:
		        break

class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "train_aug\image",
			 label_path = "train_aug\label", test_path = "test\image",
			 npy_path = "train_aug\\npydata", img_type = "png"):

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
			label = load_img(self.label_path + "\\" + midname[:-4] + "_sfa.png",grayscale = True)
			# label = load_img(self.label_path + "\\" + midname[:-4] + ".png",grayscale = True)
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
		
		# np.save('LYM_sfa\\aug_data\\train_processed1\\imgs_train_processed.npy', imgs_train)
		# imgs = np.load('LYM_sfa\\aug_data\\train_processed1\\imgs_train_processed.npy')
		# for i in range(imgs.shape[0]):
		# 	img = imgs[i]
		# 	img = array_to_img(img)
		# 	img = img.convert("RGB")
		# 	img.save("LYM_sfa\\aug_data\\train_processed1\\train_%d_processed.jpg" % (i))
		#
		# np.save('LYM_sfa\\aug_data\\train_processed1\\imgs_train_mask_processed.npy', imgs_mask_train)
		# imgs = np.load('LYM_sfa\\aug_data\\train_processed1\\imgs_train_mask_processed.npy')
		# for i in range(imgs.shape[0]):
		# 	img = imgs[i]
		# 	img = array_to_img(img)
		# 	img = img.convert("RGB")
		# 	img.save("LYM_sfa\\aug_data\\train_processed1\\train_mask_%d_processed.jpg" % (i))

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
	# augdata = myAugmentation()
	# augdata.Augmentation()
	mydata = dataProcess(512,512)
	mydata.create_train_data()
	mydata.create_test_data()
	# mydata.load_train_data()
