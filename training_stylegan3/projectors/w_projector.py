import copy
import numpy as np
import jittor as jt
from tqdm import tqdm
from configs import global_config, hyperparameters

from lpips.lpips import LPIPS

def project(
        G,
        target,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        #w_name: str
):
    #assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)
    
    G = copy.deepcopy(G)
    G.eval()

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, 512)
    w_samples = G.mapping(jt.array(z_samples), None)  # [N, L, C]
    w_samples = w_samples.numpy().astype(np.float32)
    print(w_samples.shape)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_avg_tensor = jt.array(w_avg)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    #noise_bufs = {name: buf for (name, buf) in G.named_parameters() if 'noises' in name}

    # Features for target image.
    target_images = target
    if target_images.shape[2] > 256:
        target_images = jt.nn.interpolate(target_images, size=(256, 256), mode='bilinear')

    w_opt = jt.array(start_w)  # pylint: disable=not-callable
    #print(w_opt.requires_grad)
    # Init noise.
    #for buf in noise_bufs.values():
    #    buf[:] = jt.randn_like(buf)
    #    buf.requires_grad = True
    #    print(buf.requires_grad)
    
    optimizer = jt.nn.Adam([w_opt], betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    loss_fn = LPIPS(net='vgg', spatial=False)

    for step in tqdm(range(num_steps)):
        #print(step)
        
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = jt.randn_like(w_opt) * w_noise_scale
        #w_noise.requires_grad = True
        ws = (w_opt + w_noise).repeat([1, 16, 1])
        synth_images = G.synthesis(ws)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = jt.nn.interpolate(synth_images, size=(256, 256), mode='bilinear')

        # Features for synth images.
        dist = loss_fn(target_images, synth_images)

        # Noise regularization.
        #reg_loss = 0.0
        #for v in noise_bufs.values():
        #    noise = v
        #    while True:
        #        reg_loss += (noise * jt.misc.roll(noise, shifts=1, dims=3)).mean() ** 2
        #        reg_loss += (noise * jt.misc.roll(noise, shifts=1, dims=2)).mean() ** 2
        #        if noise.shape[2] <= 8:
        #            break
        #        noise = jt.nn.avg_pool2d(noise, kernel_size=2)
        
        #loss = dist + reg_loss * regularize_noise_weight
        loss = dist

        #if step % image_log_step == 0:
        #    with torch.no_grad():
        #        if use_wandb:
        #            global_config.training_step += 1
        #            wandb.log({f'first projection _{w_name}': loss.detach().cpu()}, step=global_config.training_step)
        #            log_utils.log_image_from_w(w_opt.repeat([1, G.mapping.num_ws, 1]), G, w_name)


        optimizer.step(loss)
        #logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        #with jt.no_grad():
        #    for buf in noise_bufs.values():
        #        buf = buf - buf.mean()
        #        buf = buf * 1.0 / (buf*buf).mean().sqrt()

    del G
    return w_opt.repeat([1, 16, 1])

