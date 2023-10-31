import sys
sys.path.append('.')

import torch

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
from tqdm import tqdm

from models import SparseUNet, SparseEncoder, MLPSkipNet
from utils import load_latents
import inverse_diffusion as gd
from inverse_diffusion import get_named_beta_schedule, InverseSpacedDiffusion, space_timesteps
# from inverse_diffusion_mol_ksp import get_named_beta_schedule, InverseSpacedDiffusion, space_timesteps
from inverse_utils import A_forward, A_adjoint, nrmse, to_cplx, sampling_mask_gen
from diffusion import SpacedDiffusion
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("decoder_config", None, "Decoder training configuration.", lock_config=True)
flags.DEFINE_integer("sample_img_size", None, "The image size to sample at.")
flags.DEFINE_integer("sample_num", None, "The image to reconstruct")
flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_integer("R", None, "Accleration factor")
flags.DEFINE_float("gamma", None, "DC weighting")
flags.DEFINE_string("sampling_steps", "100", "Number of diffusion steps when sampling.")
flags.mark_flags_as_required(["config", "decoder_config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')


def main(argv):
    sample_img_size = FLAGS.sample_img_size
    sampling_steps = FLAGS.sampling_steps
    sample_num = FLAGS.sample_num
    R = FLAGS.R
    gamma = FLAGS.gamma
    H = FLAGS.config
    H.decoder = FLAGS.decoder_config
    train_kwargs = {}
    # set seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    # load kspace data

    # load sample 
    if sample_img_size<512:
        data_file = f'/home/blevac/infty-diff/fastMRI_preprocessing/test_data/sample{sample_num}_{sample_img_size}.pt'
        data_cont = torch.load(data_file)
        # print('data keys: ',data_cont.keys())
        kspace = data_cont['kspace'][0][None].cuda()
        # print('kspace shape: ',kspace.shape)
        maps = data_cont['maps'][0][None].cuda()
        # print('maps shape: ',maps.shape)
        gt_img = data_cont['gt_img'][0][None].cuda()
        # print('gt_img shape: ',gt_img.shape)
        # mask = data_cont['mask'][0][None].cuda()
        mask = torch.tensor(sampling_mask_gen(ACS_perc=0.03, R=R, img_sz=gt_img.shape[-1])).cuda()
        print('mask shape: ',mask.shape)
        mask = torch.zeros(1,1,sample_img_size,sample_img_size).cuda()
        mask[:,:,:,0::R]=1

    elif sample_img_size==512:
        data_file = '/home/blevac/infty-diff/real_data_512x512.pt'
        data_cont = torch.load(data_file)
        # print('data keys: ',data_cont.keys())
        kspace = data_cont['kspace'][0][None].cuda()
        # print('kspace shape: ',kspace.shape)
        maps = data_cont['maps'][0][None].cuda()
        # print('maps shape: ',maps.shape)
        gt_img = data_cont['gt_img'][0][None].cuda()
        # print('gt_img shape: ',gt_img.shape)
        # mask = data_cont['mask'][0][None].cuda()
        mask = torch.tensor(sampling_mask_gen(ACS_perc=0.03, R=R, img_sz=gt_img.shape[-1])).cuda()
        # print('mask shape: ',mask.shape)
        mask = torch.zeros(1,1,sample_img_size,sample_img_size).cuda()
        mask[:,:,:,0::R]=1

    scale = torch.max(abs(gt_img))
    gt_img=gt_img/scale
    kspace = kspace/scale * mask
    
    # img_size = gt_img.shape[-1]

    # get diffusion model
    img_size = sample_img_size

    results_dir='./results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir,f'sample_num{sample_num}_img_size{sample_img_size}_R{R}_gamma{gamma}_seed{FLAGS.seed}.pt')

    diff_model = SparseUNet(
            channels=H.decoder.data.channels,
            nf=H.decoder.model.nf,
            time_emb_dim=H.decoder.model.time_emb_dim,
            img_size=H.decoder.data.img_size,
            num_conv_blocks=H.decoder.model.num_conv_blocks,
            knn_neighbours=H.decoder.model.knn_neighbours,
            uno_res=H.decoder.model.uno_res,
            uno_mults=H.decoder.model.uno_mults,
            z_dim=H.decoder.model.z_dim,
            conv_type=H.decoder.model.uno_conv_type,
            depthwise_sparse=H.decoder.model.depthwise_sparse,
            kernel_size=H.decoder.model.kernel_size,
            backend=H.decoder.model.backend,
            blocks_per_level=H.decoder.model.uno_blocks_per_level,
            attn_res=H.decoder.model.uno_attn_resolutions,
            dropout_res=H.decoder.model.uno_dropout_from_resolution,
            dropout=H.decoder.model.uno_dropout,
            uno_base_nf=H.decoder.model.uno_base_channels,
            # continuous_conv=H.decoder.model.continuous_conv
        )

    diff_model_checkpoint_path = f'checkpoints/{H.decoder.run.experiment}/checkpoint.pkl'
    diff_model_state_dict = torch.load(diff_model_checkpoint_path, map_location="cpu")
    print(f"Loading diffusion model from step {diff_model_state_dict['global_step']}")
    diff_model.load_state_dict(diff_model_state_dict["model_unet_state_dict"])

    diff_model = diff_model.to(device)

   
    ## Setup diffusion for decoder model
    betas = get_named_beta_schedule(H.decoder.diffusion.noise_schedule, H.decoder.diffusion.steps, resolution=H.decoder.data.img_size)
    if H.decoder.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.decoder.diffusion.model_mean_type == "v":
        model_mean_type = gd.ModelMeanType.V
    elif H.decoder.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.decoder.diffusion.model_mean_type == "mollified_epsilon":
        assert H.decoder.diffusion.gaussian_filter_std > 0, "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception("Unknown model mean type. Expected value in [epsilon, v, xstart]")
    model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    loss_type = gd.LossType.MSE if H.decoder.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    print(sampling_steps)
    skipped_timestep_respacing = sampling_steps
    skipped_use_timesteps = space_timesteps(H.decoder.diffusion.steps, skipped_timestep_respacing)

    decoder_diffusion = InverseSpacedDiffusion(
                skipped_use_timesteps, 
                betas=betas, 
                model_mean_type=model_mean_type, 
                model_var_type=model_var_type, 
                loss_type=loss_type,
                gaussian_filter_std=H.decoder.diffusion.gaussian_filter_std,
                img_size=H.decoder.data.img_size,
                rescale_timesteps=True,
                multiscale_loss=H.decoder.diffusion.multiscale_loss, 
                multiscale_max_img_size=H.decoder.diffusion.multiscale_max_img_size,
                mollifier_type=H.decoder.diffusion.mollifier_type,
                maps = maps,
                mask = mask,
                meas = kspace,
                gamma = gamma,
                gt_img = gt_img,
            ).to(device)

    # decoder_diffusion = SpacedDiffusion(
    #         skipped_use_timesteps, 
    #         betas=betas, 
    #         model_mean_type=model_mean_type, 
    #         model_var_type=model_var_type, 
    #         loss_type=loss_type,
    #         gaussian_filter_std=H.decoder.diffusion.gaussian_filter_std,
    #         img_size=H.decoder.data.img_size,
    #         rescale_timesteps=True,
    #         multiscale_loss=H.decoder.diffusion.multiscale_loss, 
    #         multiscale_max_img_size=H.decoder.diffusion.multiscale_max_img_size,
    #         mollifier_type=H.decoder.diffusion.mollifier_type,
    #     ).to(device)

    new_std = H.decoder.diffusion.gaussian_filter_std * (sample_img_size / H.decoder.data.img_size)
    decoder_diffusion.mollifier = gd.DCTGaussianBlur(img_size, std=new_std).to(device)

    noise_mul = img_size / H.decoder.data.img_size
    idx = 0
    
    with torch.cuda.amp.autocast(enabled=H.train.amp):
        final_img, stack,_,pred_start = decoder_diffusion.ddim_sample_loop(
                                diff_model, 
                                (1, H.decoder.data.channels, img_size, img_size), 
                                progress=False, 
                                return_all=True,
                                noise_mul=noise_mul,
                                eta=0.0,
                                clip_denoised=False
                            )

    img = final_img
    img = img.detach()
    deblurred_img = decoder_diffusion.mollifier.undo_wiener(img)
    inverse_deblurred_img = decoder_diffusion.mollifier.inverse(img)
    img_out_cplx = to_cplx(img)
    deblurred_img_out_cplx = to_cplx(deblurred_img)
    inverse_deblurred_img_out_cplx = to_cplx(inverse_deblurred_img)
    img_stack_cplx = to_cplx(stack)

    mollified_nrmse = nrmse(gt_img, img_out_cplx)
    deblurred_nrmse = nrmse(gt_img, deblurred_img_out_cplx)
    inverse_nrmse = nrmse(gt_img, inverse_deblurred_img_out_cplx)

    print('mollified NRMSE: %.3f  Weiner NRMSE: %.3f  Inverse NRMSE: %.3f'%(mollified_nrmse.item(), deblurred_nrmse.item(), inverse_nrmse.item()))

    dict = {
        'gt_img':gt_img.cpu().numpy(),
        'deblurred_img_out':deblurred_img_out_cplx.cpu().numpy(),
        'inverse_deblurred_img_out':inverse_deblurred_img_out_cplx.cpu().numpy(),
        'img_out':img_out_cplx.cpu().numpy(),
        'kspace': kspace.cpu().numpy(),
        # 'img_stack': img_stack_cplx,
        'mask': mask,
    }

    torch.save(dict,results_file)

if __name__ == '__main__':
    app.run(main)