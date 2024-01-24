import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from model.model_SR_x4 import *
import argparse
import h5py
from torch.nn import functional as F
from os.path import exists, join, basename
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import os
from cal_ssim import SSIM, set_random_seed
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='BDT')

    parser.add_argument('--dataroot', type=str, default='./data/harvard_x4')
    parser.add_argument('--dataset', type=str, default='harvard_x4')
    parser.add_argument('--n_bands', type=int, default=20)
    parser.add_argument('--clip_max_norm', type=int, default=10)

    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

    # learning settingl
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='end epoch for training')
    parser.add_argument('--n_epochs', type=int, default=2001,
                        help='end epoch for training')

    args = parser.parse_args()
    return args

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        self.GT = dataset.get("GT")
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        input_rgb = torch.from_numpy(self.RGB[index, :, :, :]).float()
        input_lr = torch.from_numpy(self.LRHSI[index, :, :, :]).float()
        input_lr_u = torch.from_numpy(self.UP[index, :, :, :]).float()
        target = torch.from_numpy(self.GT[index, :, :, :]).float()

        return input_rgb, input_lr, input_lr_u, target

    #####必要函数
    def __len__(self):
        return self.GT.shape[0]


def get_test_set(root_dir):
    test_dir = join(root_dir, "test_harvardv3(with_up)x4.h5")
    return DatasetFromHdf5(test_dir)


if __name__ == '__main__':
    opt = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(opt)

    test_set = get_test_set(opt.dataroot)
    test_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=True)

    opt.n_bands = 31
    opt.image_size = 1024
    opt.n_bands_rgb = 3

    model = Bidinet(opt).cuda()
    checkpoint = torch.load('./checkpoints/BDT_harvard_x4/model_epoch_best.pth.tar')
    model.load_state_dict(checkpoint['model'].state_dict())

    psnr_ = 0
    ergas_ = 0
    pad_amount = (0, 24, 0, 24)
    pad_amount2 = (0, 6, 0, 6)

    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(test_data_loader):
            input_rgb, ms, input_lr_u, ref = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            input_rgb = F.pad(input_rgb, pad=pad_amount, mode='constant', value=0)
            ms = F.pad(ms, pad=pad_amount2, mode='constant', value=0)
            input_lr_u = F.pad(input_lr_u, pad=pad_amount, mode='constant', value=0)
            ref = F.pad(ref, pad=pad_amount, mode='constant', value=0)
            out = model(input_rgb, input_lr_u, ms)
            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            psnr = calc_psnr(ref, out)
            ergas = calc_ergas(ref, out)
            psnr_ += psnr
            ergas_ += ergas

            red = 28
            green = 14
            blue = 3

            # lr = np.squeeze(ms.detach().cpu().numpy())
            # lr_red = lr[red, :, :][:, :, np.newaxis]
            # lr_green = lr[green, :, :][:, :, np.newaxis]
            # lr_blue = lr[blue, :, :][:, :, np.newaxis]
            # lr = np.concatenate((lr_blue, lr_green, lr_red), axis=2)
            # lr = 255 * (lr - np.min(lr)) / (np.max(lr) - np.min(lr))
            # lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite('./data/{}/cave4_{}_lr.jpg'.format(opt.dataset,index), lr)
            #
            # out = np.squeeze(out)
            # out_red = out[red, :, :][:, :, np.newaxis]
            # out_green = out[green, :, :][:, :, np.newaxis]
            # out_blue = out[blue, :, :][:, :, np.newaxis]
            # out = np.concatenate((out_blue, out_green, out_red), axis=2)
            # out = 255 * (out - np.min(out)) / (np.max(out) - np.min(out))
            # cv2.imwrite('./data/{}/{}_out.jpg'.format(opt.dataset,index), out)
            #
            # ref = np.squeeze(ref)
            # ref_red = ref[red, :, :][:, :, np.newaxis]
            # ref_green = ref[green, :, :][:, :, np.newaxis]
            # ref_blue = ref[blue, :, :][:, :, np.newaxis]
            # ref = np.concatenate((ref_blue, ref_green, ref_red), axis=2)
            # ref = 255 * (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
            # cv2.imwrite('./data/{}/{}_ref.jpg'.format(opt.dataset, index), ref)
            #
            # lr_dif = np.uint8(1.5 * np.abs((lr - ref)))
            # lr_dif = cv2.cvtColor(lr_dif, cv2.COLOR_BGR2GRAY)
            # lr_dif = cv2.applyColorMap(lr_dif, cv2.COLORMAP_JET)
            # cv2.imwrite('./data/{}/{}_lr_dif.jpg'.format(opt.dataset, index), lr_dif)
            #
            # out_dif = np.uint8(1.5 * np.abs((out - ref)))
            # out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
            # out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_JET)
            # cv2.imwrite('./data/{}/{}_out_dif.jpg'.format(opt.dataset, index), out_dif)

            select_spe = 10
            out_spe = out[:, select_spe, 100:200, 100:200]
            ref_spe = ref[:, select_spe, 100:200, 100:200]
            out_spe = np.squeeze(out_spe)
            ref_spe = np.squeeze(ref_spe)

            plt.figure(figsize=(12, 8))
            plt.title('Spectral Plot for Index {}'.format(select_spe))
            plt.plot(out_spe.mean(axis=-1), label='Predicted Spectra')
            plt.plot(ref_spe.mean(axis=-1), label='Reference Spectra')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.show()

        print('PSNR:   {:.4f};'.format(psnr_ / 10))
        print('ERGAS:    {:.4f}.'.format(ergas_ / 10))


