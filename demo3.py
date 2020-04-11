import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

os.chdir("D:/code/Semantic-segmentation/2015/unet/unet-first/Lymphoma/")
in_path = '1.dcm'
out_path = 'output/output.jpg'


def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height


def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    information['NumberOfFrames'] = ds.NumberOfFrames
    return information

def showImage(img_array, frame_num=0):
    img_bitmap = Image.fromarray(img_array[frame_num])
    return img_bitmap

def limitedEqualize(img_array, limit=4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized


def writeVideo(img_array,filename):
    frame_num, width, height = img_array.shape
    filename_output = filename.split('.')[0] + '.avi'
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
    for img in img_array:
        video.write(img)
    video.release()


img_array, frame_num, width, height = loadFile(in_path)
img_array_limited_equalized = limitedEqualize(img_array)
# show
plt.imshow(img_array_limited_equalized) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

# save
# 适用于保存任何 matplotlib 画出的图像，相当于一个 screencapture
plt.savefig('fig_dog.png')
# cv2.imwrite("1.jpg",img_array_limited_equalized)