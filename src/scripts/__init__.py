import numpy as np
import torch
import torchvision
import PIL
import os


def save_video(img,outdir, drange,fname="video0.mp4", normalize=True):
    _, C ,T ,H ,W = img.shape
    # print (f'Saving Video with {T} frames, img shape {H}, {W}')

    img = img.cpu().detach().numpy()
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)
    # gw, gh = grid_size
    # _N, C, T, H, W = img.shape
    # img = img.reshape(gh, gw, C, T, H, W)
    img = np.squeeze(img)
    img = img.transpose(1,2,3,0)
    # img = img.reshape(T, H,  W, C)
    # assert C in [3]
    if C == 3:
        torchvision.io.write_video(os.path.join(outdir,fname), torch.from_numpy(img), fps=8)
        # imgs = [PIL.Image.fromarray(img, 'RGB') for i in range(len(img))]
        # imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)
