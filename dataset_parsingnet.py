from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy
import glob

class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        # 返回指定路径下的文件和文件夹列表。
        self.args = args

        if args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4_bicubic')
            self.imgs_parsing_path = os.path.join(root, 'global_32（Gound Ture）')
        elif args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR')
            self.imgs_parsing_path = os.path.join(root, 'LR_x4_bicubic')
        elif args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16')
            self.imgs_parsing_path = os.path.join(root, 'global_2_LR_x16')
        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.jpg'))
        )
        self.imgs_parsing = sorted(
            glob.glob(os.path.join(self.imgs_parsing_path, '*.jpg'))
        )
        self.transform = transforms.ToTensor()
        self.train = train

    def __getitem__(self, item):

        img_path_LR = os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_parsing = os.path.join(self.imgs_parsing_path, self.imgs_parsing[item])
        LR = Image.open(img_path_LR)
        parsing = Image.open(img_path_parsing)
        LR = numpy.array(LR)
        parsing = numpy.array(parsing)
        LR = ToTensor()(LR)
        parsing = ToTensor()(parsing)
        filename = os.path.basename(img_path_LR)

        return LR, parsing, filename


    def __len__(self):
        return len(self.imgs_LR)

