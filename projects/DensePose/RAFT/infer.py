import sys
sys.path.append('core')

from PIL import Image
import argparse
import os, pdb, imageio, tqdm
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

@torch.no_grad()
def infer(model, seq_img_dir, suffix, iters=24, backward_flow=True):
    if backward_flow:
        flow_img_dir = os.path.join(seq_img_dir, '../flow_backward_img_{}'.format(suffix))
        flow_np_dir = os.path.join(seq_img_dir, '../flow_backward_np_{}'.format(suffix))
        # flow_np_save_path = os.path.join(seq_img_dir, '../flow_backward_{}.npy'.format(suffix))
    else:
        flow_img_dir = os.path.join(seq_img_dir, '../flow_forward_img_{}'.format(suffix))
        flow_np_dir = os.path.join(seq_img_dir, '../flow_forward_np_{}'.format(suffix))
        # flow_np_save_path = os.path.join(seq_img_dir, '../flow_forward_{}.npy'.format(suffix))
    if not os.path.exists(flow_img_dir):
        os.makedirs(flow_img_dir)
    if not os.path.exists(flow_np_dir):
        os.makedirs(flow_np_dir)

    model.eval()
    dataset = datasets.InferVideoDataset(seq_img_dir, backward_flow=backward_flow)

    # flow_list, flow_img_list = [], []
    for val_id in tqdm.tqdm(range(len(dataset))):
        image1, image2, path1, path2 = dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='sintel')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        # map flow to rgb image
        # pdb.set_trace()
        # flow = flow[0].permute(1,2,0).cpu().numpy()
        flow = flow.permute(1,2,0).cpu().numpy()
        flow_img = flow_viz.flow_to_image(flow)

        # flow_list.append(flow)
        # flow_img_list.append(flow_img)
        imageio.imwrite(os.path.join(flow_img_dir, path1.split('/')[-1]), flow_img)
        np.save(os.path.join(flow_np_dir, path1.split('/')[-1].split('.')[0]+'.npy'), flow)
        # del image1, image2, flow, flow_img

    # flow_array = np.concatenate(flow_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--seq_img_dir', help="sequence images for evaluation")
    parser.add_argument('--backward_flow', action='store_true', help='calculate flow from i+1 to i')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        suffix = args.model.split('-')[-1].split('.')[0]
        infer(model.module, args.seq_img_dir, suffix)


