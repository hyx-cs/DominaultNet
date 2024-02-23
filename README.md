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
```
  Note: Here the processed images of CelebA are divided into three folders named train/test/val as training set/test set/validation set respectively.
```
  ### Instructions
  * Change this to the path of the CelebA image (option.py)
  ```
  parser.add_argument('--dir_data', type=str, default='D:\\dataset\\CelebA\\img_align_celeba\\',
                    help='dataset directory')
  ```
  * When training the network, modify the following code to the corresponding paths. (dataset_parsing.py)
  ```
        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR_8_bicubic')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_16_bicubic')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_4_bicubic')
                        ···
        self.imgs_parsing_path = os.path.join(root, 'global_2')
  ```
  ```
  Note: There should be corresponding HR/LR/Global/ folders in the three files (train/test/val) for storing the ground truth/input images of different sizes/prior images.
  ```

## Test
In the testing phase, pre-trained models provided by the project can be used, as well as models that are additionally trained.
