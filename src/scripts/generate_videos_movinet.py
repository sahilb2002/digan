"""Generates a dataset of images using pretrained network pickle."""
import math
import sys; sys.path.extend(['.', 'src'])
import os
import random

# import click
import numpy as np
import torch

# from scripts import save_video
from einops import rearrange
import tensorflow as tf
from scripts import *


torch.set_grad_enabled(False)


# @click.command()
# @click.pass_context
# @click.option('--network_pkl', help='Network pickle filename', required=True)
# @click.option('--timesteps', type=int, help='Timesteps', default=16, show_default=True)
# @click.option('--num_videos', type=int, help='Number of images to generate', default=100, show_default=True)
# @click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_videos_movinet(
    network_pkl: str,
    timesteps: int,
    num_videos: int,
    seed: int,
    outdir: str,
    model,
    label_path,
    G
):
    # print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    #     G.forward = Generator.forward.__get__(G, Generator)
    #     print("Done. ")

    os.makedirs(outdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    grid_z = torch.randn([int(num_videos), G.z_dim], device=device).split(1)

    images = [rearrange(
                        G(z, None, timesteps=timesteps, noise_mode='const')[0].cuda(),
                        '(b t) c h w -> b c t h w', t=timesteps) for z in grid_z]    
    out_tens=[]
    out_lab = []
    f = open(label_path,'r')
    label_map = list(f.read().split('\n'))
    # print("video0 = ")
    # print(images[0])
    for img in images:
        save_video(img, outdir, drange=[-1, 1],fname="video0.mp4")
        input_tensor = img.detach().cpu().numpy()
        input_tensor = input_tensor.transpose(0,2,3,4,1)
        input_tensor = input_tensor.astype('float32')
        input_tensor = (input_tensor-input_tensor.min())/(input_tensor.max()-input_tensor.min())
        # print(input_tensor.max(),input_tensor.min())
        input_tensor = tf.convert_to_tensor(input_tensor)
        output = model(input_tensor)
        label_ind = tf.argmax(output, -1)[0].numpy()
        img = img.transpose(1,2)
        out_tens.append(input_tensor)
        out_lab.append(label_ind)
    
    return out_tens,out_lab


if __name__ == "__main__":
    generate_videos() 
