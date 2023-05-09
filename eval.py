import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder, calculate_psnr
from data import test_dataloader
from utils import EvalTimer
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import sys, scipy.io

def _eval(model, config):
    model_pretrained = os.path.join('results/', config.model_name, config.test_model)
    state_dict = torch.load(model_pretrained)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img = data

            input_img, label_img = data
            input_img = input_img.to(device)
            
            tm = time.time()
            pred = model(input_img)
            elaps = time.time() - tm
            adder(elaps)

            p_numpy = pred[config.num_subband].squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            ssim = structural_similarity(p_numpy, in_numpy, multichannel=True, data_range=1, win_size=3)
            # ... the rest of the code remains the same

            psnr_adder(psnr)
            ssim_adder(ssim)
            print('%d iter PSNR: %.2f SSIM: %.4f time: %f' % (iter_idx + 1, psnr, ssim, elaps))

        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f' % (ssim_adder.average()))
        print("Average time: %f"%adder.average())
