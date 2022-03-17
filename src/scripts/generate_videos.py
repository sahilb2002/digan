"""Generates a dataset of images using pretrained network pickle."""
import math
import sys; sys.path.extend(['.', 'src'])
import os
import random

# import click
import dnnlib
import numpy as np
import torch

import legacy
from training.networks import Generator
from scripts import save_video
from einops import rearrange
from mmaction.apis import inference_recognizer,init_recognizer

torch.set_grad_enabled(False)


# @click.command()
# @click.pass_context
# @click.option('--network_pkl', help='Network pickle filename', required=True)
# @click.option('--timesteps', type=int, help='Timesteps', default=16, show_default=True)
# @click.option('--num_videos', type=int, help='Number of images to generate', default=100, show_default=True)
# @click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_videos(
    network_pkl: str,
    timesteps: int,
    num_videos: int,
    seed: int,
    outdir: str,
    model,
    label_path,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
        G.forward = Generator.forward.__get__(G, Generator)
        print("Done. ")

    os.makedirs(outdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    grid_z = torch.randn([int(num_videos), G.z_dim], device=device).split(1)

    images = [rearrange(
                        G(z, None, timesteps=timesteps, noise_mode='const')[0].cuda(),
                        '(b t) c h w -> b c t h w', t=timesteps) for z in grid_z]    
    out=[]
    f = open(label_path,'r')
    label_map = list(f.read().split('\n'))

    for img in images:
        save_video(img, outdir, drange=[-1, 1],fname="video0.mp4")
        label = max(inference_recognizer(model,os.path.join(outdir,"video0,mp4"),label_path),key=lambda item:item[1])[0]
        out.append((img,label_map.index(label)))
    
    return out


if __name__ == "__main__":
    generate_videos()
