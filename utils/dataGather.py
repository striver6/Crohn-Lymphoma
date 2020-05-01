import os
images = []
out_file = open('D:/code/Semantic-segmentation/2015/unet/unet-first/Lymphoma/images.txt','w')
for root, dirs, files in os.walk('L:/data/医学/Lymphoma', topdown=False):
    for name in files:
        if name.endswith('.png'):
            # print(os.path.join(root, name))
            l = os.path.join(root, name)
            print(str(l))
            images.append(os.path.join(root, name))
            out_file.write(str(l) + '\n')
        # images.append([os.path.join(root, name) for name in files if name.endswith('.png')])
print(images[0].split('\\')[-2])
