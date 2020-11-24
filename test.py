

import torch
from torch.autograd import Variable

import os 
import sys
from os import mkdir
from os.path import isfile, isdir
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('-n', type=int, default=2, help='Divisor, make larger when GPU memory shortage')
parser.add_argument('-p', '--path', type=str, required=True)
args = parser.parse_args()


## Model
import RRDBNet_arch as arch

model_G = arch.RRDBNetx4x4(3, 3, 64, 23, gc=32).cuda()
model_G.load_state_dict(torch.load('./model.pth').state_dict(), strict=True)
model_G.eval()


## Create output folder
if not isdir('./output'):
    mkdir('./output')


## Test
with torch.no_grad():

    # Input images
    files = glob.glob(f'{args.path}/*.png')
    files.sort()
    dirname = os.path.dirname(args.path).split("/")
    output_path = os.path.join("output", dirname[3], dirname[-1])
    os.makedirs(output_path, exist_ok=True)

    for fn in tqdm(files):
        # Load images
        img = Image.open(fn)
        img = np.asarray(img).astype(np.float32) / 255.0 # HxWxC
        img = np.transpose(img, [2, 0, 1]) # CxHxW
        img = img[np.newaxis, ...] # BxCxHxW
        batch_input = Variable(torch.from_numpy(img), volatile=True).cuda()

        # divide and conquer strategy due to GPU memory limit
        _, _ , H, W = batch_input.size()

        dh = args.n
        dw = args.n
        i_pad = 2

        hs = []
        for h in range(dh):
            sh = (H//dh)*h - i_pad
            eh = (H//dh)*(h+1) + i_pad
            hh_pad = [i_pad*16, i_pad*16]
            if sh < 0:
                sh = 0
                hh_pad[0] = 0
            if eh > H:
                eh = H
                hh_pad[1] = 0

            ws = []
            for w in range(dw):
                sw = (W//dw)*w - i_pad
                ew = (W//dw)*(w+1) + i_pad
                ww_pad = [i_pad*16, i_pad*16]
                if sw < 0:
                    sw = 0
                    ww_pad[0] = 0
                if ew > W:
                    es = W
                    ww_pad[1] = 0

                slice_input = batch_input[:,:, sh:eh, sw:ew]
                slice_output = model_G(slice_input)

                #
                slice_output = (slice_output).cpu().data.numpy()
                slice_output = np.clip(slice_output[0], 0. , 1.)
                slice_output = np.transpose(slice_output, [1, 2, 0])

                if hh_pad[0] > 0:
                    slice_output = slice_output[hh_pad[0]:,:,:]
                if hh_pad[1] > 0:
                    slice_output = slice_output[:-hh_pad[1],:,:]
                if ww_pad[0] > 0:
                    slice_output = slice_output[:,ww_pad[0]:,:]
                if ww_pad[1] > 0:
                    slice_output = slice_output[:,:-ww_pad[1],:]

                ws.append(slice_output)

            hs.append( np.concatenate(ws, 1) )
        batch_output = np.concatenate(hs, 0)

        # Save to file
        Image.fromarray(np.around(batch_output*255).astype(np.uint8)).save(os.path.join(output_path, os.path.basename(fn)))


