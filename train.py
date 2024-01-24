import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import dataset_parsing
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import net
import util
import loss
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
epochs = args.epochs
lr = args.lr
net = net.net()
net = util.prepare(net)

print(util.get_parameter_number(net))
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)

traindata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
trainset = DataLoader(traindata, batch_size=16, shuffle=True, num_workers=0)
valdata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
valset = DataLoader(valdata, batch_size=4, shuffle=False, num_workers=0)

for i in range(epochs):
    net.train()
    train_loss = 0
    bum = len(trainset)
    for batch, (lr, hr, parsing, _) in enumerate(trainset):
        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        sr = net(lr, parsing)
        # lr, hr = util.prepare(lr), util.prepare(hr)
        # sr = net(lr)
        sr = sr[0]

        loss = criterion1(sr, hr)
        train_loss = train_loss + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch：{} loss: {:.3f}".format(i + 1, train_loss / (len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss / (len(trainset)) * 255, i + 1)
    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, '../model_train'), exist_ok=True)
    torch.save(net.state_dict(),
               os.path.join(args.save_path, args.writer_name, '../model_train', 'epoch{}.pth'.format(i + 1)))
    net.eval()
    val_psnr = 0
    val_ssim = 0
    img_grid = []
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)

    for batch, (lr, hr, parsing, _) in enumerate(valset):
        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        sr = net(lr, parsing)

        # lr, hr = util.prepare(lr), util.prepare(hr)
        # sr = net(lr)
        sr = sr[0]
        psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())

        val_psnr = val_psnr + psnr_c
        val_ssim = val_ssim + ssim_c
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        # 将多张图片拼接成一张图片，中间用黑色网格分割
        # create grid of images
        img_grid = torchvision.utils.make_grid(sr)
        writer.add_image('image_grid', img_grid)

    print("Epoch：{} val  psnr: {:.3f}".format(i + 1, val_psnr / (len(valset))))
    print("Epoch：{} val  ssim: {:.3f}".format(i + 1, val_ssim / (len(valset))))
    writer.add_scalar("val_psnr_DIC", val_psnr / len(valset), i + 1)
    writer.add_scalar("val_ssim_DIC", val_ssim / len(valset), i + 1)
