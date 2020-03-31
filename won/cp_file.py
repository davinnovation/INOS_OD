import os

source_f_path = "/media/cvpr-miu/4TB_2/data/ImageNet_cropped/train/"
dest_f_path = '/media/cvpr-miu/4TB_2/data/VOC_ImageNet/train/'

save_class = []
file_name = ""

f = open("class.txt", "r")
file_name = f.readlines()

for f_name in file_name:
    save_class.append(f_name[:-1])
    
print(save_class)

f.close()

for c_name in save_class:
    os.system('cp -r {}{} {}{}'.format(source_f_path, c_name, dest_f_path, c_name))
