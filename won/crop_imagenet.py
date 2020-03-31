import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

from PIL import Image
from matplotlib import pyplot as plt
#import cv2
import os
from tqdm import tqdm

import argparse

import time
start = time.time()

#torch.backends.cudnn.benchmark=True
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataroot', type=str, default='/media/cvpr-miu/4TB_2/data/ImageNet/train',
                   help='Root directory of ImageNet data set')
parser.add_argument('--saveroot', type=str, default='/media/cvpr-miu/4TB_2/data/ImageNet_cropped/train',
                   help='Saving root directory of preprocessed images')
parser.add_argument('--resize', type=int, default=256,
                   help='Resizing size')
parser.add_argument('--p_start', type=int, default=10,
                   help='Percentage of initial cropping size 10 20...100 possible')
parser.add_argument('--p_end', type=int, default=100,
                   help='Percentage of initial cropping size 10 20...100 possible')
parser.add_argument('--batch_size', type=int, default=60,
                   help='Batch size for the images at loading')
parser.add_argument('--num_workers', type=int, default=6,
                   help='Number of threads to use at dataloader')
parser.add_argument('-d','--debug', action = "store_true")

args = parser.parse_args()

if args.debug:
  import pdb
  pdb.set_trace()

data_root = args.dataroot
save_root = args.saveroot
resize = args.resize
p_start = args.p_start
p_end = args.p_end
batch_size = args.batch_size
num_workers = args.num_workers

#for checking variables
print("Data root : ", data_root)
print("Save root : ", save_root)
print("Resize size : ", resize)
print("Percentage of start cropping size : ", p_start)
print("Percentage of end cropping size : ", p_end)
print("Batch size : ", batch_size)
print("Number of workers : ", num_workers)

#tensor to PIL Image
to_pil = transforms.ToPILImage()

if not os.path.exists(save_root):
        os.makedirs(save_root)

f = open(os.path.join(save_root, 'train_cropped_annotation.txt'), 'w')

#get path from each images
class ImageFolderWithPaths(dset.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

#cosntruct transform
def make_transforms(percentage):
    make_transform = transforms.Compose([transforms.Resize((resize,resize)),
                                   transforms.CenterCrop(resize*0.1*percentage),
                                   transforms.Resize((resize,resize)),
                                   transforms.ToTensor()])
    return make_transform

def make_saving_folder(target_dir):
    #path = os.path.join(save_root, target_dir)
    path = save_root + '/' + target_dir
    if not os.path.exists(path):
        os.makedirs(path)

    return path
    
if __name__ == '__main__':

    processed_images = 0

    for percentage in range(int(p_start/10), int(p_end/10)+1):
        dataset = ImageFolderWithPaths(root=data_root, transform=make_transforms(percentage))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers)

        # iterate over data
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels, paths = data
            count_images = len(paths)

            for cnt in range(count_images):
                #convert tensor to image
                img_c = to_pil(inputs[cnt])

                #for each class folder, save to the new folder named same as class folder
                #if new folder doesn't exist, create and return path
                class_file = paths[cnt].split('/')[-2]
                new_save_path = make_saving_folder(class_file)

                img_file = os.path.splitext(os.path.basename(paths[cnt]))[0]

                # plt.imshow(img_c)
                img_c.save(os.path.join(new_save_path, "{0}__{1:0>3}.jpg".format(img_file, int(percentage*10))))
                f.write(str(new_save_path + "/{0}__{1:0>3}.jpg\n".format(img_file, int(percentage*10))))
                processed_images += 1

                #if(processed_images % 1000 == 0):
                #    print("Number of processed images : ", processed_images)

    print("Number of total processed images : ", processed_images)

f.close()
end = time.time()
print("Processed time(sec) : ", end - start)
