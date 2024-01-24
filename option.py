import argparse
import os

parser = argparse.ArgumentParser(description='FaceSR')

parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--cuda_name', type=str, default='0')
parser.add_argument('--gpu_ids', type=int, default=1)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--dir_data', type=str, default='D:\\dataset\\CelebA\\img_align_celeba\\',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--data_val', type=str, default='val',
                    help='val dataset name')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--augment', action='store_true',
                    help='use data augmentation')

# Model specifications###################################
parser.add_argument('--model_train', default='net',
                    help='model_train name')
#########################################################

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')
parser.add_argument('--large', action="store_true",
                    help='the input is as large as output or not')

# Training specifications
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
##########################################################################
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
##########################################################################
parser.add_argument('--test_only', action='store_true',# default=True,
                    help='set this option to test the model_train')
parser.add_argument('--l1_weight', type=float, default=1,
                    help='loss function configuration')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')

# Log specifications
parser.add_argument('--save_path', type=str, default='./experiment',
                    help='file path to save model_train')
parser.add_argument('--parsing_load', type=str, default='C:\\Users\\12140\\Desktop\\code\\Pretrain_model\\Parsing_model.pth',
                    help='file name to load')
parser.add_argument('--load', type=str, default='C:\\Users\\12140\\Desktop\\code\\Pretrain_model\\Pretrain_model.pth',
                    help='file name to load')

parser.add_argument("--writer_name", type=str, default="test_result",
                    help="the name of the writer")
parser.add_argument("--writer_name1", type=str, default="model_parsing",
                    help="the name of the writer")

args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

