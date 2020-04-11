import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import os
import numpy as np


os.chdir("L:/data/医学/Lymphoma/caixieyang/")
in_path = 'original/1.dcm'
out_path = 'output/output.jpg'
scans = []
# ds = pydicom.read_file(in_path,force=True)  #读取.dcm文件
ds = pydicom.dcmread(in_path,force=True)
print(ds.WindowCenter)
print(ds.WindowWidth)
# for i in len(ds.WindowWidth):
#     center = ds.WindowCenter(i) /ds.RescaleSlope - ds.RescaleIntercept;
#
#     width = ds.WindowWidth(i) / ds.RescaleSlope - ds.RescaleIntercept;
#
#     M = mat2gray(ds, [center - (width / 2), center + (width / 2)]);

    # a = strcat(in_path, num2str(i));
    #
    # a = strcat(a, '.bmp'); % 输出文件名的操作
    #
    # imwrite(M, a, 'bmp'); % 输出为文件

    # subplot(1, lop, i), imshow(a), title(a); % 显示
ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

# img = ds.pixel_array  # 提取图像信息
# print(img.shape)
# plt.imshow(img)
# plt.show()
# scipy.misc.imsave(out_path,img)  # 2018/11/6更新，之前保存图片的方式出了一点小问题，对不住，对不住。
#
# scans.append(ds)
#
# def get_pixels_hu(scans):
#     # type(scans[0].pixel_array)
#     # Out[15]: numpy.ndarray
#     # scans[0].pixel_array.shape
#     # Out[16]: (512, 512)
#     # image.shape: (129,512,512)
#     image = np.stack([s.pixel_array for s in scans])
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#
#     # Set outside-of-scan pixels to 1
#     # The intercept is usually -1024, so air is approximately 0
#     image[image == -2000] = 0
#
#     # Convert to Hounsfield units (HU)
#     intercept = scans[0].RescaleIntercept
#     slope = scans[0].RescaleSlope
#
#     if slope != 1:
#         image = slope * image.astype(np.float64)
#         image = image.astype(np.int16)
#
#     image += np.int16(intercept)
#
#     return np.array(image, dtype=np.int16)
#
# print(get_pixels_hu(scans).shape)

