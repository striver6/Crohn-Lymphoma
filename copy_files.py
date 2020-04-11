import shutil
import os
os.chdir("D:\\code\\Semantic-segmentation\\2015\\unet\\unet-first\\LYM_vfa")
datasets_path = "LYM_vfa"+"\\"
old_aug_train_path="aug_merge"
old_aug_label_path="aug_label"
aug_train_path= "aug_data\\image"
aug_label_path= "aug_data\\label"

# for root, dirs, files in os.walk(old_aug_train_path, topdown=False):
#     for file in files:
#         shutil.copy(os.path.join(root, file), aug_train_path)

len1 = []
for root, dirs, files in os.walk(old_aug_train_path, topdown=False):
    for dir in dirs:
        for files in os.walk(os.path.join(root,dir)):
            len1.append(len(files[2]))

# for root, dirs, files in os.walk(old_aug_label_path, topdown=False):
#     for file in files:
#         shutil.copy(os.path.join(root, file), aug_label_path)

len2 = []
for root, dirs, files in os.walk(old_aug_label_path, topdown=False):
    for dir in dirs:
        for files in os.walk(os.path.join(root, dir)):
            len2.append(len(files[2]))


print(len(len1))
print(len(len2))
for i in range(len(len1)):
    if(len2[i]!=len1[i]):
        print(i,"--",len2[i]==len1[i],"--",len1[i],len2[i])