from unet import *
from data import *
import matplotlib.pylab as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start = datetime.now()

mydata = dataProcess(512,512)
imgs_test = mydata.load_test_data()
imgs_train,imgs_mask_train = mydata.load_train_data()
imgs_train = imgs_train
print(imgs_train.shape)
print(imgs_test.shape)
myunet = myUnet()
model = myunet.get_unet()
# model.load_weights('unethaha.hdf5')
# model.load_weights('LYM_sfa\\unet.hdf5')
# model.load_weights('LYM_sfa/aug_data/unet-2019-12-28-23-10-132-0.45.hdf5')
# model.load_weights('LYM_sfa/unet3.hdf5')
# model.load_weights('LYM_sfa/unet3.hdf5')
# model.load_weights('LYM_sfa/unet-2020-01-01-16-00-09-0.96-0.88.hdf5')
# model.load_weights('LYM_sfa/unet-2020-01-03-22-31-29-0.93-0.89.hdf5')
model.load_weights('LYM_sfa/results6/unet-2020-01-16-18-29-100-0.92-0.95.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size=1,verbose=1)
imgs_mask_train = model.predict(imgs_train, batch_size=1,verbose=1)
print("imgs_train.shape:",imgs_train.shape)
print("imgs_mask_train.shape:",imgs_mask_train.shape)
print('*'*100)
print(imgs_mask_test.shape)
# nums = imgs_mask_test.shape[0]
# height = imgs_mask_test.shape[1]
# weight = imgs_mask_test.shape[2]
# channels = imgs_mask_test.shape[3]
# for num in range(nums):            #遍历高
#     for row in range(height):            #遍历高
#         for col in range(weight):         #遍历宽
#             for c in range(channels):     #便利通道
#                 if(imgs_mask_test[num,row, col, c]>0):
#                     # print("num:",num)
#                     imgs_mask_test[num,row, col, c] = imgs_mask_test[num,row, col, c]*255
print(imgs_mask_test.max())
print(imgs_mask_test.min())
print(imgs_mask_test.mean())




np.save('LYM_sfa\\results6\\imgs_mask_test_pred.npy', imgs_mask_test)
print("array to image")
imgs = np.load('LYM_sfa\\results6\\imgs_mask_test_pred.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    # print(img)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img.point(lambda i: i*2000)
    # # print(img)
    # print('+'*100)
    # print(img.info)
    # print(img.size)
    # print(img.mode)
    # plt.imshow(img)
    # plt.show()
    img.save("LYM_sfa\\results6\\test_%d_pred.jpg" % (i))
    # input("hit enter to cont : ")
#
#
# np.save('LYM_sfa\\results6\\imgs_mask_train_pred.npy', imgs_mask_train)
np.save('LYM_sfa\\results6\\imgs_mask_train_pred.npy', imgs_mask_train)
print("array to image")
# imgs = np.load('LYM_sfa\\results6\\imgs_mask_train_pred.npy')
imgs = np.load('LYM_sfa\\results6\\imgs_mask_train_pred.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    # print(img)
    img = array_to_img(img)
    img = img.convert("RGB")
    img = img.point(lambda i: i*2000)
    # # print(img)
    # print('+'*100)
    # print(img.info)
    # print(img.size)
    # print(img.mode)
    # plt.imshow(img)
    # plt.show()
    # img.save("LYM_sfa\\results6\\train_%d_pred.jpg" % (i))
    img.save("LYM_sfa\\results6\\train_%d_pred.jpg" % (i))

stop = datetime.now()
print(stop - start)







