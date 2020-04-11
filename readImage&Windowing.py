import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import os
import numpy as np
import SimpleITK as sitk

# os.chdir("D:/code/Semantic-segmentation/2015/unet/unet-first/Lymphoma/")

import os
dicoms = []
for root, dirs, files in os.walk('D:/code/Semantic-segmentation/2015/unet/unet-first/Lymphoma/original', topdown=False):
# for root, dirs, files in os.walk('D:/code/Semantic-segmentation/2015/unet/unet-first/Lymphoma/testdcm', topdown=False):
    for name in files:
        if name.endswith('.dcm'):
            # print(os.path.join(root, name))
            dicoms.append(os.path.join(root, name))
    # images.append([os.path.join(root, name) for name in files if name.endswith('.png')])
print(len(dicoms))
print(dicoms)


def get_pixels_hu(scans):
    # type(scans[0].pixel_array)
    # Out[15]: numpy.ndarray
    # scans[0].pixel_array.shape
    # Out[16]: (512, 512)
    # image.shape: (129,512,512)
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# print(get_pixels_hu(scans).shape)

def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


# for i in range(len(images)):
#     print("第",i,"张")
#     ds = pydicom.dcmread(images[i],force=True)
#     ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
#     for j in range(len(ds.WindowCenter)):
#         img = ds.pixel_array  # 提取图像信息
#         img1 = transform_ctdata(img, ds.WindowWidth[j],ds.WindowCenter[j])
#         scipy.misc.imsave('images/{}_{}_1.png'.format(images[i].split('\\')[-3],i), img1)

#
# for i in images:
# center = ds.WindowCenter[1]/ds.RescaleSlope - ds.RescaleIntercept
# width = ds.WindowWidth[1]/ds.RescaleSlope - ds.RescaleIntercept


# size_file = open('Lymphoma/train/files.txt', 'w')
for i in range(len(dicoms)):
    in_path = dicoms[i]
    # in_path = '1.dcm'
    # out_path = 'output.jpg'
    scans = []
    # ds = pydicom.read_file(in_path,force=True)  #读取.dcm文件
    seg = sitk.ReadImage(in_path)
    ds = pydicom.dcmread(in_path, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    img = ds.pixel_array  # 提取图像信息

    # size_file.write(in_path + '---' + str(i) + '.png' + '\n')
    # scipy.misc.imsave('Lymphoma/train/images/{}.png'.format(i), img)


    center = ds.WindowCenter[1]
    width = ds.WindowWidth[1]

    # print(ds.WindowWidth)
    # print(ds.WindowCenter)
    # print(width)
    # print(center)
    img1 = transform_ctdata(img, 310, 1020)
    # scipy.misc.imsave('output2/{}_{}_2.jpg'.format(width,center),img1)
    # scipy.misc.imsave('Lymphoma/output4/{}_{}_{}.jpg'.format(j,1020,i), img1)
    scipy.misc.imsave('Lymphoma/output4/{}.jpg'.format(i), img1)



# print(img.shape)
# plt.imshow(img)
# plt.show()
# scipy.misc.imsave(out_path,img)  # 2018/11/6更新，之前保存图片的方式出了一点小问题，对不住，对不住。
#
# scans.append(ds)

