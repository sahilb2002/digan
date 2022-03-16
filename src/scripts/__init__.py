import numpy as np
import torch
import torchvision
import PIL
import os


def save_video_grid(img,outdir, drange, normalize=True):
    _, C ,T ,H ,W = img[0].shape
    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    for i in range(len(img)):
        img[i] = img[i].numpy()
        if normalize:
            lo, hi = drange
            img[i] = np.asarray(img[i], dtype=np.float32)
            img[i] = (img[i] - lo) * (255 / (hi - lo))
            img[i] = np.rint(img[i]).clip(0, 255).astype(np.uint8)
        # gw, gh = grid_size
        # _N, C, T, H, W = img.shape
        # img = img.reshape(gh, gw, C, T, H, W)
        img[i] = np.squeeze(img[i])
        img[i] = img[i].transpose(1,2,3,0)
        # img[i] = img[i].reshape(T, H,  W, C)
        # assert C in [3]
        if C == 3:
            torchvision.io.write_video(os.path.join(outdir,f'video{i}.mp4'), torch.from_numpy(img[i]), fps=8)
            # imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
            # imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)
