import os

#查找文件
normal_path="D:/code/Semantic-segmentation/2015/unet/unet-first/temp1/"
normal_result_path="D:/code/Semantic-segmentation/2015/unet/unet-first/temp2/"



#os.listdir()方法，列出来所有文件
#返回path指定的文件夹包含的文件或文件夹的名字的列表
# print(normal_result_path.split('/')[-0]+".txt")
def rename_file(path,result_path):
    #主逻辑
    files = os.listdir(path)
    #对于批量的操作，使用FOR循环
    index = 216
    # print(result_path.split('/')[-0]+".txt")
    # outfile = open(result_path.split('/')[-0]+".txt",'w')
    outfile = open("temp3/files1.txt",'w')
    files.sort(key=lambda x: int(x[:-9]))
    for f in files:
        #调试代码的方法：关键地方打上print语句，判断这一步是不是执行成功
        print(f)
        old_file = os.path.join(path, f)
        # new_file=os.path.join(result_path,str(index)+".png")
        new_file=os.path.join(result_path,""+str(index)+".png")
        # new_file=os.path.join(result_path,f+"_mask.png")
        # print(f,"---",new_file)
        # print("File will be renamed as:{}".format(new_file))
        os.rename(old_file,new_file)
        # print("修改后的文件名是:{}".format(f))
        index += 1
        outfile.write(f + "---" + new_file + "\n")

rename_file(normal_path,normal_result_path)





# import shutil
#
# for root, dirs, files in os.walk(old_path, topdown=False):
#     for file in files:
#         if(file.endswith(".png")):
#             shutil.copy(os.path.join(root, file), new_path)