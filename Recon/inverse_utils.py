import torch
import torch.fft as torch_fft
import torch.nn as nn
import random
import numpy as np
# import sigpy as sp


def nrmse(x, y):
    num = torch.norm(x-y, p=2)
    denom = torch.norm(x,p=2)
    return num/denom



def nrmse_np(x, y):
    num = np.linalg.norm(x-y)
    denom = np.linalg.norm(x)
    return num/denom

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * \
                  np.conj(s_maps), axis=1)

def A_forward(image, maps, mask):
    #image shape: [B,1,H,W]
    #maps shape: [B,C, H,W]
    # mask shape: [B,1,H,W]

    coil_imgs = maps*image
    coil_ksp = fft(coil_imgs)
    sampled_ksp = mask*coil_ksp
    return sampled_ksp

def A_adjoint(ksp, maps, mask):
    # ksp shape: [B,1,H,W]
    # maps shape: [B,C, H,W]
    # mask shape: [B,1,H,W]

    sampled_ksp = mask*ksp
    coil_imgs = ifft(sampled_ksp)
    img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1) #sum over coil dimension

    return img_out[:,None,...]

def to_cplx(img):
    ''' Convert real image to complex image '''
    # assumes img input shape is [B,2,H,W]
    cplx_img = img[:,0,...] + 1j*img[:,1,...]
    return cplx_img[:,None,...]
def to_real(img):
    ''' Convert real image to complex image '''
    # assumes img input shape is [B,1,H,W]
    real_img = torch.view_as_real(img[:,0,...]).permute(0,-1,1,2)
    
    return real_img

def sampling_mask_gen(ACS_perc, R, img_sz):
    ACS = int(ACS_perc*img_sz)
    outer_line_count = int(int(img_sz/R) - ACS)
    mask = np.zeros((img_sz,img_sz))
    # use 14 lines for center of ksp
    ACS_start = int(img_sz/2)-int(ACS/2)
    ACS_end = ACS_start + ACS
    center_idx = np.arange(ACS_start,ACS_end) # ACS_perc*img_sz central lines for ACS
    total_idx = np.arange(img_sz)
    rem_lines = np.delete(total_idx, center_idx)

    random.shuffle(rem_lines)
    mask_lines=np.concatenate((center_idx,rem_lines[0:outer_line_count]))


    # print(mask_lines)
    mask[:,mask_lines] = 1

    mask_batched = mask[None,None]

    return mask_batched
