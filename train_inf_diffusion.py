import torch
from einops import rearrange
import numpy as np
from scipy.stats.qmc import Halton

import wandb
# import visdom
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import time
import copy
import os
from tqdm import tqdm

# from model import Unet
from models import SparseUNet
# from model_transformer_v2 import Transformer
from utils import get_data_loader, flatten_collection, optim_warmup, \
    plot_images,plot_images_cplx, update_ema, create_named_schedule_sampler, LossAwareSampler
import diffusion as gd
from diffusion import GaussianDiffusion, get_named_beta_schedule

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

# Torch options
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device = torch.device('cuda')

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Tracking functions
def track_variable(logs, variable, name):
    logs['track_'+name] = np.append(logs['track_'+name], variable.item())

def update_logs_means(logs):
    for name in list(logs.keys()):
        if 'track_' in name:
            mean_name = 'mean_'+name[6:]
            logs[mean_name] = np.mean(logs[name])
            logs[name] = np.array([])
    
from torch.profiler import profile, record_function, ProfilerActivity

def train(H, model, ema_model, train_loader, test_loader, optim, diffusion, schedule_sampler, vis=None, checkpoint_path='', global_step=0):
    halton = Halton(2)
    scaler = torch.cuda.amp.GradScaler()

    mean_loss = 0
    mean_step_time = 0
    mean_total_norm = 0
    skip = 0
    while True:
        for x in train_loader:
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            start_time = time.time()

            if global_step < H.optimizer.warmup_steps:
                optim_warmup(global_step, optim, H.optimizer.learning_rate, H.optimizer.warmup_steps)

            global_step += 1
            x = x.to(device, non_blocking=True)
            if H.data.channels!=2: # if data is not MRI then do the normal resacaling stuff
                x = x * 2 - 1 

            t, weights = schedule_sampler.sample(x.size(0), device)

            # TODO: Try low discrepancy sequence from "Discretization Invariant Learning on Neural Fields"
            if H.mc_integral.type == 'uniform':
                sample_lst = torch.stack([torch.from_numpy(np.random.choice(H.data.img_size**2, H.mc_integral.q_sample, replace=False)) for _ in range(H.train.batch_size)]).to(device)
            elif H.mc_integral.type == 'halton':
                # TODO: Make this completely continuous. Use grid_sample. And use continuous coordinate embeddings
                sample_lst = torch.stack([torch.from_numpy((halton.random(H.mc_integral.q_sample) * H.data.img_size).astype(np.int64)) for _ in range(H.train.batch_size)]).to(device) # BxLx2
                sample_lst = sample_lst[:,:,0] * H.data.img_size + sample_lst[:,:,1]
            else:
                raise Exception('Unknown Monte Carlo Integral type')

            with torch.cuda.amp.autocast(enabled=H.train.amp):
                # with profile(activities=[
                #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
                losses = diffusion.training_losses(model, x, t, sample_lst=sample_lst)
                if H.diffusion.multiscale_loss:
                    loss = (losses["multiscale_loss"] * weights).mean()
                else:
                    loss = (losses["loss"] * weights).mean()
            
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            # exit()
            
            optim.zero_grad()
            if H.train.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if H.optimizer.gradient_skip and model_total_norm >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    scaler.step(optim)
                    scaler.update()
            else:
                loss.backward()
                model_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if H.optimizer.gradient_skip and model_total_norm >= H.optimizer.gradient_skip_threshold:
                    skip += 1
                else:
                    optim.step()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            if global_step % H.train.ema_update_every == 0:
                update_ema(model, ema_model, H.train.ema_decay)
            
            mean_loss += loss.item()
            mean_step_time += time.time() - start_time
            mean_total_norm += model_total_norm.item()

            wandb_dict = dict()
            if global_step % H.train.plot_graph_steps == 0 and global_step > 0:
                norm = H.train.plot_graph_steps
                print(f"Step: {global_step}, Loss {mean_loss / norm:.5f}, Step Time: {mean_step_time / norm:.5f}, Skip: {skip / norm:.5f}, Gradient Norm: {mean_total_norm / norm:.5f}")
                wandb_dict |= {'Step Time': mean_step_time / norm, 'Loss': mean_loss / norm, 'Skip': skip / norm, "Gradient Norm": mean_total_norm / norm}
                mean_loss = 0
                mean_step_time = 0
                skip = 0
                mean_total_norm = 0

            # if global_step % H.train.plot_recon_steps == 0 and global_step > 0:
            #     plot_images(H, x, title='x', vis=vis)
            #     plot_images(H, losses["x_t"], title='x_noisy', vis=vis)

                # if "pred_xstart" in losses:
                #     plot_images(H, losses["pred_xstart"], title='recon', vis=vis)
            
            if global_step % H.train.plot_samples_steps == 0 and global_step > 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=H.train.amp):
                        samples, _ = diffusion.p_sample_loop(ema_model, (H.train.sample_size, H.data.channels, H.data.img_size, H.data.img_size), progress=True)


                if H.data.channels == 2:
                    wandb_dict |= plot_images_cplx(H, samples, title='samples', vis=vis) # special ploting for complex images
                    
                else:
                    wandb_dict |= plot_images(H, samples, title='samples', vis=vis)

                if H.diffusion.model_mean_type == "mollified_epsilon":
                    if H.data.channels == 2:
                        wandb_dict |= plot_images_cplx(H, diffusion.mollifier.undo_wiener(samples), title=f'deblurred_samples', vis=vis) # special ploting for complex images
                    else:
                        wandb_dict |= plot_images(H, diffusion.mollifier.undo_wiener(samples), title=f'deblurred_samples', vis=vis)
            
            # if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
            #     # Rough approximation of test loss to assess for overfitting. TODO: Run whole loop.
            #     total_loss, count = 0.0, 0
            #     for x in tqdm(test_loader):
            #         if isinstance(x, tuple) or isinstance(x, list):
            #             x = x[0]
            #         x = x.to(device)
            #         for _ in range(H.train.test_loss_repeats):
            #             t, weights = schedule_sampler.sample(x.size(0), device)
            #             with torch.no_grad():
            #                 with torch.cuda.amp.autocast(enabled=H.train.amp):
            #                     losses = diffusion.training_losses(ema_model, x, t)
            #                     loss = (losses["loss"] * weights).mean()
            #             total_loss += loss.item()
            #             count += 1
            #     print(f"Test Loss: {total_loss/count}")
            #     wandb_dict |= {'Test Loss': total_loss/count}


            if wandb_dict:
                wandb.log(wandb_dict, step=global_step)
            
            if global_step % H.train.checkpoint_steps == 0 and global_step > 0:
                torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'model_unet_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        # 'scheduler_state_dict': schedule.state_dict()
                    }, checkpoint_path)

                

def main(argv):
    H = FLAGS.config
    train_kwargs = {}

    # wandb can be disabled by passing in --config.run.wandb_mode=disabled
    wandb.init(project=H.run.name, config=flatten_collection(H), save_code=True, dir=H.run.wandb_dir, mode=H.run.wandb_mode)
    # if H.run.enable_visdom:
    #     train_kwargs['vis'] = visdom.Visdom(server=H.run.visdom_server, port=H.run.visdom_port)
    

    model = SparseUNet(
        channels=H.data.channels,
        nf=H.model.nf,
        time_emb_dim=H.model.time_emb_dim,
        img_size=H.data.img_size,
        num_conv_blocks=H.model.num_conv_blocks,
        knn_neighbours=H.model.knn_neighbours,
        uno_res=H.model.uno_res,
        uno_mults=H.model.uno_mults,
        z_dim=H.model.z_dim,
        conv_type=H.model.uno_conv_type,
        depthwise_sparse=H.model.depthwise_sparse,
        kernel_size=H.model.kernel_size,
        backend=H.model.backend,
        blocks_per_level=H.model.uno_blocks_per_level,
        attn_res=H.model.uno_attn_resolutions,
        dropout_res=H.model.uno_dropout_from_resolution,
        dropout=H.model.uno_dropout,
        uno_base_nf=H.model.uno_base_channels
    )
    # NOTE: deepcopy doesn't work on Minkowksi
    ema_model = SparseUNet(
        channels=H.data.channels,
        nf=H.model.nf,
        time_emb_dim=H.model.time_emb_dim,
        img_size=H.data.img_size,
        num_conv_blocks=H.model.num_conv_blocks,
        knn_neighbours=H.model.knn_neighbours,
        uno_res=H.model.uno_res,
        uno_mults=H.model.uno_mults,
        z_dim=H.model.z_dim,
        conv_type=H.model.uno_conv_type,
        depthwise_sparse=H.model.depthwise_sparse,
        kernel_size=H.model.kernel_size,
        backend=H.model.backend,
        blocks_per_level=H.model.uno_blocks_per_level,
        attn_res=H.model.uno_attn_resolutions,
        dropout_res=H.model.uno_dropout_from_resolution,
        dropout=H.model.uno_dropout,
        uno_base_nf=H.model.uno_base_channels
    )
    # else:
    #     model = UNet(
    #         channels=H.data.channels,
    #         nf=H.model.nf,
    #         img_size=H.data.img_size,
    #         num_conv_blocks=H.model.num_conv_blocks,
    #         knn_neighbours=H.model.knn_neighbours,
    #         uno_res=H.model.uno_res,
    #         uno_mults=H.model.uno_mults,
    #         uno_modes=H.model.uno_modes,
    #         z_dim=H.model.z_dim,
    #         conv_type=H.model.uno_conv_type,
    #         depthwise_sparse=H.model.depthwise_sparse
    #     )
    #     ema_model = copy.deepcopy(model)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Load checkpoints
    if H.run.experiment != '':
        checkpoint_path = f'checkpoints/{H.run.experiment}/'
    else:
        checkpoint_path = 'checkpoints/'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = checkpoint_path + 'checkpoint.pkl'
    train_kwargs['checkpoint_path'] = checkpoint_path

    model = model.to(device)
    ema_model = ema_model.to(device)
    train_loader, test_loader = get_data_loader(H)
    optim = torch.optim.Adam(
            model.parameters(), 
            lr=H.optimizer.learning_rate, 
            betas=(H.optimizer.adam_beta1, H.optimizer.adam_beta2)
        )

    if H.train.load_checkpoint and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loading Model from step {state_dict['global_step']}")
        train_kwargs['global_step'] = state_dict['global_step']
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        ema_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
        try:
            optim.load_state_dict(state_dict['optimizer_state_dict'])
        except ValueError:
            print("Failed to load optim.")

    # timestep_respacing = [H.diffusion.steps]
    # use_timesteps = space_timesteps(H.diffusion.steps, timestep_respacing)
    # betas = get_named_beta_schedule(H.diffusion.noise_schedule, H.diffusion.steps)
    # model_mean_type=(gd.ModelMeanType.EPSILON if not H.diffusion.predict_xstart else gd.ModelMeanType.START_X)
    # model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    # loss_type = gd.LossType.MSE if H.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    
    betas = get_named_beta_schedule(H.diffusion.noise_schedule, H.diffusion.steps, resolution=H.data.img_size)
    if H.diffusion.model_mean_type == "epsilon":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif H.diffusion.model_mean_type == "v":
        model_mean_type = gd.ModelMeanType.V
    elif H.diffusion.model_mean_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif H.diffusion.model_mean_type == "mollified_epsilon":
        assert H.diffusion.gaussian_filter_std > 0, "Error: Predicting mollified_epsilon but gaussian_filter_std == 0."
        model_mean_type = gd.ModelMeanType.MOLLIFIED_EPSILON
    else:
        raise Exception("Unknown model mean type. Expected value in [epsilon, v, xstart]")
    model_var_type=((gd.ModelVarType.FIXED_LARGE if not H.model.sigma_small else gd.ModelVarType.FIXED_SMALL) if not H.model.learn_sigma else gd.ModelVarType.LEARNED_RANGE)
    loss_type = gd.LossType.MSE if H.diffusion.loss_type == 'mse' else gd.LossType.RESCALED_MSE
    diffusion = GaussianDiffusion(
            betas, 
            model_mean_type, 
            model_var_type, 
            loss_type, 
            H.diffusion.gaussian_filter_std, 
            H.data.img_size,
            rescale_timesteps=True, 
            multiscale_loss=H.diffusion.multiscale_loss, 
            multiscale_max_img_size=H.diffusion.multiscale_max_img_size,
            mollifier_type=H.diffusion.mollifier_type
        ).to(device)

    schedule_sampler = create_named_schedule_sampler(H.diffusion.schedule_sampler, diffusion)

    train(H, model, ema_model, train_loader, test_loader, optim, diffusion, schedule_sampler, **train_kwargs)

if __name__ == '__main__':
    app.run(main)
