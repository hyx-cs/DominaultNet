from option import args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import dataset_parsing
from torch.utils.data import DataLoader
import util
import torchvision
from model import net

epochs = args.epochs
net = net.net()
net = util.prepare(net)
print(util.get_parameter_number(net))

testdata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=2, shuffle=False, num_workers=0)
pretrained_dict = torch.load(args.load, map_location='cuda:0')
net.load_state_dict(pretrained_dict)
net = util.prepare(net)
net.eval()

# teacher.eval()
val_psnr = 0
val_ssim = 0
with torch.no_grad():
    save_name = "result-test"
    if "helen" in args.dir_data:
        save_name = 'result-test-helen'
    elif "FFHQ" in args.dir_data:
        save_name = 'result-test-FFHQ'
    elif "CelebA" in args.dir_data:
        save_name = 'result'
    else:
        save_name = "result-test"

    os.makedirs(os.path.join(args.save_path, args.writer_name, save_name), exist_ok=True)
    net.eval()
    for batch, (lr, hr, parsing, filename) in enumerate(testset):
        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        sr = net(lr, parsing)
        sr = sr[0]
        psnr1, ssim1 = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu(), crop_border=8)
        val_psnr = val_psnr + psnr1
        val_ssim = val_ssim + ssim1

        torchvision.utils.save_image(sr[0],
                                     os.path.join(args.save_path, args.writer_name, save_name,
                                                  '{}'.format(str(filename[0])[:-4] + ".jpg")))
    print("Test psnr: {:.3f}".format(val_psnr / (len(testset))))
    print("Test ssim: {:.3f}".format(val_ssim / (len(testset))))

