# DominaultNet
> Faces super resolution, Multiscale Parsing Attention Module, Multichannel Detail Attention Module

## Framework
<div align=center><img width="760" height="340" src=img/img_1.jpg/></div>

## Prerequisites

- Windows 
- Python 3.10
- Pytorch 2.0.1
- NVIDIA GPU + CUDA


## Train
* Firstly, you need to download the CelebA dataset and crop the center of the CelebA image to 128*128 size.
* Secondly, after downsampling the image to 16*16 size, it will be used as ground truth and input image respectively.
* Finally, set the paths of ground truth and input image in the corresponding positions.

:star: Note: Here the processed images of CelebA are divided into three folders named train/test/val as training set/test set/validation set respectively.

  ### Instructions
  * Change this to the path of the CelebA image (option.py)
  ```
  parser.add_argument('--dir_data', type=str, default='Folder path of CelebA image',
                    help='dataset directory')
  ```
  * When training the parsing network, modify the following code to the corresponding paths. (dataset_parsingnet.py)
  ```
        if args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
            self.imgs_parsing_path = os.path.join(root, 'The prior image generation from ground truth')
        elif args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
            self.imgs_parsing_path = os.path.join(root, 'The prior image generation from ground truth')
        elif args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
            self.imgs_parsing_path = os.path.join(root, 'The prior image generation from ground truth')
  ```

  * When training the network, modify the following code to the corresponding paths. (dataset_parsing.py)
  ```
        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR folder path for CelebA images')
                        ···
        self.imgs_parsing_path = os.path.join(root, 'parsing folder path for CelebA images')
  ```
:star: Note: There should be corresponding HR/LR/Global/ folders in the three files (train/test/val) for storing the ground truth/input images of different sizes/prior images.

## Test
In the testing phase, pre-trained models provided by the project can be used, as well as models that are additionally trained.
```
parser.add_argument('--save_path', type=str, default='./experiment',
                    help='file path to save model_train')
parser.add_argument('--parsing_load', type=str, default='The training model of the parsingnet',
                    help='file name to load')
parser.add_argument('--load', type=str, default='The training model of the network',
                    help='file name to load')
```

## Citation
If you use this code for your research, please cite our paper.
> Yongxi Hu, Liang Chen, Zezhao Su, Yaoguo Shen, Na Qi. FACE SUPER-RESOLUTION VIA MULTI-SCALE LOW RESOLUTION PRIOR AND DUAL ATTENTION NETWORKS. IEEE International Conference on Image
Processing

## Acknowledgments
Our code is inspired by [FishFSRNet](https://github.com/wcy-cs/FishFSRNet) and [SCET](https://github.com/AlexZou14/SCET).
