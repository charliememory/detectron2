# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os, pdb, random
import hashlib
import zipfile
from six.moves import urllib
import numpy as np
from PIL import Image
import imageio
## Ref: https://github.com/matplotlib/matplotlib/issues/9294#issuecomment-469235699
## To solve ERROR: Unexpected segmentation fault
import matplotlib.pyplot as plt

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
    
def disp2colormap_single(d, color_map_name="gnuplot"):
    """Change gray to color map
    """
    assert 3==len(d.shape)
    # if 1==d.shape[0]:
    #     d = torch.cat([d]*3, dim=0)
    # pdb.set_trace()
    d = normalize_image(d)
    if type(d) is np.ndarray:
        # H x W x C
        d = d[:,:,0]
    else:
        # C x H x W
        d = d.permute(1,2,0).detach().cpu().numpy()[:,:,0]
    if color_map_name=="gnuplot":
        cmap = plt.cm.gnuplot
    elif color_map_name=="gnuplot2":
        cmap = plt.cm.gnuplot2
    elif color_map_name=="hot":
        cmap = plt.cm.hot
    elif color_map_name=="gray":
        cmap = plt.cm.gray
    norm = plt.Normalize(vmin=d.min(), vmax=d.max())
    d = cmap(norm(d))[:,:,:3] ## get RGB from RGBA
    return d

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    if type(x) is np.ndarray:
        ma = float(x.max())
        mi = float(x.min())
    else:
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


####### utils for RAFT #######
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# def forward_interpolate(flow):
#     flow = flow.detach().cpu().numpy()
#     dx, dy = flow[0], flow[1]

#     ht, wd = dx.shape
#     x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

#     x1 = x0 + dx
#     y1 = y0 + dy
    
#     x1 = x1.reshape(-1)
#     y1 = y1.reshape(-1)
#     dx = dx.reshape(-1)
#     dy = dy.reshape(-1)

#     valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
#     x1 = x1[valid]
#     y1 = y1[valid]
#     dx = dx[valid]
#     dy = dy[valid]

#     flow_x = interpolate.griddata(
#         (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

#     flow_y = interpolate.griddata(
#         (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

#     flow = np.stack([flow_x, flow_y], axis=0)
#     return torch.from_numpy(flow).float()


def grid_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, align_corners=False)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def make_meshgrid(B, H, W, norm=True):
    ygrid, xgrid = torch.meshgrid(torch.arange(H), torch.arange(W))

    # coords = torch.stack(coords[::-1], dim=0).float()
    # pdb.set_trace()
    if norm:
        xgrid = 2*xgrid.float()/(W-1) - 1
        ygrid = 2*ygrid.float()/(H-1) - 1
    grid = torch.stack([xgrid, ygrid], dim=0)
    return grid[None].repeat(B, 1, 1, 1)


# def upflow8(flow, mode='bilinear'):
#     new_size = (8 * flow.shape[2], 8 * flow.shape[3])
#     return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=False)


####### utils for img2tex #######
def resize_iuv_tensor(iuv, height, width):
    assert len(iuv.shape)==4
    I, UV = iuv[:,:-2], iuv[:,-2:]
    # pdb.set_trace()
    I = F.interpolate(I, size=(height,width), mode='nearest')
    UV = F.interpolate(UV, size=(height,width), mode='bilinear', align_corners=False)
    return torch.cat([I,UV], dim=1)


backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(tenFlow.device)

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)


def sample_img_from_tex(tex_tensor, iuv_tensor, class_num=25, tex_patch_size=256, img_space_size=512):
    b, c, h, w = iuv_tensor.shape
    assert c==class_num+2, "input I channel should be onehot encoded"
    assert tex_tensor.shape[0]==class_num-1
    assert iuv_tensor[:,-2:].min()>=0 and iuv_tensor[:,-2:].max()<=1, "input UV channels should be normalized into [0,1]"
    ## pad texture to equal size of image for sampling
    # tex_tensor = tex_tensor.reshape([class_num-1,-1,tex_tensor.shape[1],tex_tensor.shape[2]]).permute([0,3,1,2])
    # h,w = tex_tensor.shape[2], tex_tensor.shape[3]
    b_t, c_t, h_t, w_t = tex_tensor.shape
    pad_bottom = torch.zeros([b_t,c_t,h-h_t,w_t])
    pad_right = torch.zeros([b_t,c_t,h,w-w_t])
    tex_tensor = torch.cat([torch.cat([tex_tensor,pad_bottom],dim=2), pad_right], dim=3).to(iuv_tensor.device)

    ## sample each body patch from texture, and then compose them into one body image
    # out_tensor = torch.zeros([1,3,img_space_size,img_space_size]).cuda()
    for PartInd in range(1,class_num):
        prob = iuv_tensor[:,PartInd:PartInd+1] 
        ## create flow
        part_mask = (prob>0).float()
        uv = part_mask * iuv_tensor[:,-2:]
        uv *= float(tex_patch_size)/float(img_space_size)
        uv = uv*2. - 1 ## [0,1] -> [-1,1]
        # uv = uv/127.5 - 1
        if uv.max()>1 or uv.min()<-1:
            pdb.set_trace()
        # out_patch = F.grid_sample(input=tex_tensor[PartInd-1:PartInd].float(), grid=uv.permute(0, 2, 3, 1).float(), mode='bilinear', padding_mode='zeros', align_corners=False)
        out_patch = F.grid_sample(input=tex_tensor[PartInd-1:PartInd].float(), grid=uv.permute(0, 2, 3, 1).float())
        # pdb.set_trace()
        # data_path='/esat/dragon/liqianma/datasets/Pose/youtube/youtube_single'
        # imageio.imwrite(os.path.join(data_path,"out_patch{}.png".format(PartInd)), out_patch[0].permute([1,2,0]).detach().cpu().numpy())
        # imageio.imwrite(os.path.join(data_path,"tex_patch{}.png".format(PartInd)), tex_tensor[PartInd-1:PartInd].float()[0].permute([1,2,0]).detach().cpu().numpy())
        if PartInd==1:
            out_tensor = out_patch * prob
        else:
            out_tensor += out_patch * prob
    return out_tensor
    # out = out_tensor[0].permute([1,2,0]).detach().cpu().numpy()
    # return out

import softsplat
def img2tex_forwardwarp(img_tensor, iuv_tensor, class_num=25, tex_patch_size=256, img_space_size=512, inpaint_func=None, warp_mask=True, use_prob=False):
    # pdb.set_trace()
    b, c, h, w = img_tensor.shape
    assert iuv_tensor.shape[1]==class_num+2, "input I channel should be onehot encoded"
    assert iuv_tensor[:,-2:].min()>=0 and iuv_tensor[:,-2:].max()<=1, "input UV channels should be normalized into [0,1]"
    TextureIm  = torch.zeros([b,24,c,tex_patch_size,tex_patch_size], dtype=torch.float64).to(iuv_tensor.device)
    if warp_mask:
        TextureMask  = torch.zeros([b, 24,1,tex_patch_size,tex_patch_size], dtype=torch.float64).to(iuv_tensor.device)
    for PartInd in range(1,class_num):    ## Set to xrange(1,23) to ignore the face part.
        prob = iuv_tensor[:,PartInd:PartInd+1] 
        if use_prob:
            ## create flow
            part_mask = (prob>1./class_num).float()
            # pdb.set_trace()
            uv = part_mask * iuv_tensor[:,-2:]
            uv = uv*(tex_patch_size-1)
            if isinstance(img_space_size, tuple):
                coords = make_meshgrid(b, img_space_size[0], img_space_size[1], norm=False).to(iuv_tensor.device)
            else:
                coords = make_meshgrid(b, img_space_size, img_space_size, norm=False).to(iuv_tensor.device)
            flow = uv - coords
            # pdb.set_trace()
            ## create img
            img_masked = img_tensor * prob
            ## forward warp
            tex_patch = softsplat.FunctionSoftsplat(tenInput=img_masked.float(), tenFlow=flow.float(), tenMetric=None, strType='average')
            # pdb.set_trace()
            # imageio.imwrite("../tmp/tmp_tex_patch{}.png".format(PartInd), tex_patch[0].permute([1,2,0]).detach().cpu().numpy())
            tex_patch = tex_patch[:,:,:tex_patch_size,:tex_patch_size]
            if inpaint_func is not None:
                tex_patch = inpaint_func(tex_patch)
            # pdb.set_trace()
            TextureIm[:, PartInd-1] += tex_patch
            # pdb.set_trace()
            if warp_mask:
                prob = softsplat.FunctionSoftsplat(tenInput=prob, tenFlow=flow.float(), tenMetric=None, strType='average')
                prob = prob[:,:,:tex_patch_size,:tex_patch_size]
                if inpaint_func is not None:
                    prob = inpaint_func(prob)
                TextureMask[:, PartInd-1] += prob #(prob>0).float()
        else:
            # assert prob
            ## create flow
            part_mask = (prob>0).float()
            uv = part_mask * iuv_tensor[:,-2:]
            uv = uv*(tex_patch_size-1)
            if isinstance(img_space_size, tuple):
                coords = make_meshgrid(b, img_space_size[0], img_space_size[1], norm=False).to(iuv_tensor.device)
            else:
                coords = make_meshgrid(b, img_space_size, img_space_size, norm=False).to(iuv_tensor.device)
            flow = uv - coords
            # pdb.set_trace()
            ## create img
            img_masked = img_tensor * prob
            ## forward warp
            tex_patch = softsplat.FunctionSoftsplat(tenInput=img_masked.float(), tenFlow=flow.float(), tenMetric=None, strType='average')
            # pdb.set_trace()
            # imageio.imwrite("../tmp/tmp_tex_patch{}.png".format(PartInd), tex_patch[0].permute([1,2,0]).detach().cpu().numpy())
            tex_patch = tex_patch[:,:,:tex_patch_size,:tex_patch_size]
            if inpaint_func is not None:
                tex_patch = inpaint_func(tex_patch)
            # pdb.set_trace()
            TextureIm[:, PartInd-1] += tex_patch
            # pdb.set_trace()
            if warp_mask:
                prob = softsplat.FunctionSoftsplat(tenInput=prob.float(), tenFlow=flow.float(), tenMetric=None, strType='average')
                prob = prob[:,:,:tex_patch_size,:tex_patch_size]
                if inpaint_func is not None:
                    prob = inpaint_func(prob)
                TextureMask[:, PartInd-1] += prob #(prob>0).float()

        ## vis
        # flow_img = flow_viz.flow_to_image(flow[0].permute([1,2,0]).detach().cpu().numpy())
        # imageio.imwrite('flow_img.png', flow_img)
        # imageio.imwrite('u_masked.png', uv[0,0].detach().cpu().numpy())
        # imageio.imwrite('part_mask.png', part_mask[0,0].detach().cpu().numpy())
        # imageio.imwrite('img_tensor.png', img_tensor[0].permute([1,2,0]).detach().cpu().numpy())
        # imageio.imwrite('img_masked.png', img_masked[0].permute([1,2,0]).detach().cpu().numpy())
    if warp_mask:
        return TextureIm.float(), TextureMask.float()
    else:
        return TextureIm.float()

# dilate_kernel=None
def torch_inpaint_oneChannel(tensor, kernel_size=5, valid_threshold=1, dilate_kernel=None):
    if 4==len(tensor.shape):
        assert 1==tensor.shape[1]
    else:
        assert 3==len(tensor.shape)
        tensor = tensor.unsqueeze(1)
    
    # tensor = torch.clamp(tensor, -1, 1)
    # tensor = tensor*0.5 + 0.5  ## norm [-1,1] to [0,1]
    valid_idx = tensor > 0
    # tensor_valid_binary = (valid_idx > 0).float() * 1
    tensor_valid_binary = (valid_idx > 0).float() * (valid_idx <= 1).float()

    # if not hasattr(self, "dilate_kernel"):
    #     self.dilate_kernel = torch.Tensor(np.ones([kernel_size,kernel_size])).unsqueeze(0).unsqueeze(0).to(tensor.device)
    # dilate_kernel = self.dilate_kernel
    if dilate_kernel is None:
        dilate_kernel = torch.Tensor(np.ones([kernel_size,kernel_size])).unsqueeze(0).unsqueeze(0).to(tensor.device)
    
    # pad = nn.ReflectionPad2d(int(kernel_size//2))
    pad = torch.nn.ZeroPad2d(int(kernel_size//2))
    tensor_inpaint = torch.nn.functional.conv2d(pad(tensor), dilate_kernel, padding=0)
    tensor_valid_binary_inpaint = torch.nn.functional.conv2d(pad(tensor_valid_binary), dilate_kernel, padding=0)

    # valid_inpaint_idx = tensor_valid_binary_inpaint>0
    tensor_inpaint = tensor_inpaint/(tensor_valid_binary_inpaint+1e-7)
    tensor_inpaint = tensor_valid_binary*tensor +  (1-tensor_valid_binary)*tensor_inpaint

    threshold = valid_threshold
    mask = (tensor_valid_binary_inpaint>=threshold).float()
    tensor_inpaint = tensor_inpaint * mask

    # tensor_inpaint = (tensor_inpaint-0.5)/0.5
    return tensor_inpaint, dilate_kernel


def img_iuv_resize_with_center_pad(img_tensor, iuv_tensor, class_num=25, size=512):
    b, c, h, w = iuv_tensor.shape
    assert c==class_num+2
    # ratio = max(size/h,size/w)
    if h>=w:
        h_resize = size
        w_resize = float(w)/float(h)*float(size)
        h_resize, w_resize = int(h_resize), int(w_resize)
        img_tensor = F.interpolate(img_tensor, size=(h_resize, w_resize), mode='bilinear', align_corners=False)
        iuv_tensor = resize_iuv_tensor(iuv_tensor, h_resize, w_resize)
        
        w_pad_left = int(float(h_resize-w_resize)/2.)
        w_pad_right = h_resize-w_resize-w_pad_left
        with torch.no_grad():
            pad_left = torch.zeros([b,3,h_resize,w_pad_left]).to(iuv_tensor.device)
            pad_right = torch.zeros([b,3,h_resize,w_pad_right]).to(iuv_tensor.device)
        img_tensor = torch.cat([pad_left, img_tensor, pad_right], dim=3)
        with torch.no_grad():
            pad_left = torch.zeros([b,class_num+2,h_resize,w_pad_left]).to(iuv_tensor.device)
            pad_right = torch.zeros([b,class_num+2,h_resize,w_pad_right]).to(iuv_tensor.device)
        iuv_tensor = torch.cat([pad_left, iuv_tensor, pad_right], dim=3)
    else:
        h_resize = float(h)/float(w)*float(size)
        w_resize = size
        h_resize, w_resize = int(h_resize), int(w_resize)
        img_tensor = F.interpolate(img_tensor, size=(h_resize, w_resize), mode='bilinear', align_corners=False)
        iuv_tensor = resize_iuv_tensor(iuv_tensor, h_resize, w_resize)
        
        h_pad_top = int(float(w_resize-h_resize)/2.)
        h_pad_bottom = w_resize-h_resize-h_pad_top
        with torch.no_grad():
            pad_top = torch.zeros([b,3,h_pad_top,w_resize]).to(iuv_tensor.device)
            pad_bottom = torch.zeros([b,3,h_pad_bottom,w_resize]).to(iuv_tensor.device)
        # pdb.set_trace()
        img_tensor = torch.cat([pad_top, img_tensor, pad_bottom], dim=2)
        with torch.no_grad():
            pad_top = torch.zeros([b,class_num+2,h_pad_top,w_resize]).to(iuv_tensor.device)
            pad_bottom = torch.zeros([b,class_num+2,h_pad_bottom,w_resize]).to(iuv_tensor.device)
        iuv_tensor = torch.cat([pad_top, iuv_tensor, pad_bottom], dim=2)
    return img_tensor, iuv_tensor

def img_resize_with_center_crop(img_tensor, tgt_h, tgt_w):
    size = img_tensor.shape[-1]
    if tgt_h>=tgt_w:
        img_tensor = F.interpolate(img_tensor, size=(tgt_h, tgt_h), mode='bilinear', align_corners=False)
        w1 = int(float(tgt_h-tgt_w)/2.)
        img_tensor = img_tensor[:,:,:,w1:w1+tgt_w]
    else:
        img_tensor = F.interpolate(img_tensor, size=(tgt_w, tgt_w), mode='bilinear', align_corners=False)
        h1 = int(float(tgt_w-tgt_h)/2.)
        img_tensor = img_tensor[:,:,h1:h1+tgt_h,:]
    return img_tensor


def SIUV_logit_to_IonehotUV_batch(iuvlogit_batch, bbox_xywh, norm=True, prob=True):
    assert len(iuvlogit_batch.shape)==4

    b, h, w, c = iuvlogit_batch.shape
    iuv_list = []
    for ii in range(b):
        iuvlogit = iuvlogit_batch[ii:ii+1]

        S,I,U,V = iuvlogit[:,:2], iuvlogit[:,2:27], iuvlogit[:,27:52], iuvlogit[:,52:77]
        if prob:
            # S_onehot = F.softmax(S, dim=1)
            # fg_mask = torch.argmax(S, dim=1).float()[:,None,...]
            fg_mask = F.softmax(S, dim=1)[:,1:2].detach()
            I = F.softmax(I, dim=1)
        else:
            fg_mask = torch.argmax(S, dim=1).float()[:,None,...]
            num_classes = I.shape[1]
            I = torch.argmax(I, dim=1)
            # pdb.set_trace()
            I = F.one_hot(I, num_classes=num_classes).permute(0,3,1,2)
            # S_onehot = F.gumbel_softmax(S, tau=0.1, hard=True, dim=1)
            # I_onehot = F.gumbel_softmax(I, tau=0.1, hard=True, dim=1)
            # fg_mask = S_onehot[:,1:2]
            
        valid_mask = torch.zeros_like(iuvlogit[:,:1]).to(iuvlogit_batch.device)
        x, y, w, h = bbox_xywh
        # valid_mask[:,:,y:y+h,x:x+w] = 1
        valid_mask[:,:,y+1:y+h,x+1:x+w] = 1 ## +1 to avoid broad line
        # imageio.imwrite("tmp/fg_mask0.png", fg_mask[0].permute([1,2,0]).detach().cpu().numpy())
        fg_mask *= valid_mask
        # imageio.imwrite("tmp/fg_mask1.png", fg_mask[0].permute([1,2,0]).detach().cpu().numpy())
        # pdb.set_trace()

        I = I * fg_mask
        U = I*U
        U = U.sum(dim=1, keepdim=True)
        V = I*V
        V = V.sum(dim=1, keepdim=True)

        U = U.clamp(0,1)*255
        V = V.clamp(0,1)*255
        if norm:
            # I_onehot = I_onehot/24.
            U = U/255.
            V = V/255.
        iuv_list.append(torch.cat([I,U,V], dim=1))

    iuv_batch = torch.cat(iuv_list, dim=0)

    return iuv_batch, fg_mask

def SIUV_logit_to_iuv_batch(iuvlogit_batch, norm=False, use_numpy=False):
    """
    Convert DensePose outputs to results format. Results are more compact,
    but cannot be resampled any more
    """
    assert len(iuvlogit_batch.shape)==4

    if not use_numpy:
        assert isinstance(iuvlogit_batch, torch.Tensor)
        device = iuvlogit_batch.device
        iuvlogit_batch = iuvlogit_batch.permute([0,2,3,1]).detach().cpu().numpy()

    iuv_list = []
    b, h, w, c = iuvlogit_batch.shape
    for ii in range(b):
        iuvlogit = iuvlogit_batch[ii]
        S, I, U, V = iuvlogit[...,:2], iuvlogit[...,2:27], iuvlogit[...,27:52], iuvlogit[...,52:77]
        S = np.argmax(S, axis=-1)
        I = np.argmax(I, axis=-1)
        I = I * S

        UV = np.zeros([h, w, 2])
        # assign UV from channels that correspond to the labels
        for part_id in range(1, U.shape[-1]):
            UV[...,0][I == part_id] = U[..., part_id][I == part_id]
            UV[...,1][I == part_id] = V[..., part_id][I == part_id]


        # confidences = {}
        # bbox_xywh = [0,0,iuvlogit.shape[-1],iuvlogit.shape[-2]]
        # pred_densepose = DensePoseOutput(S, I, U, V, confidences)
        # ## For debug
        # # pdb.set_trace()
        # # IUV = torch.cat([I.float()/24.*255, data[:2]*255], dim=0)
        # # imageio.imwrite('tmp/IUV_wConfidence.png', IUV.permute([1,2,0]).detach().cpu().numpy())
        # # pdb.set_trace()
        # I, data = resample_output_to_bbox(pred_densepose, bbox_xywh, confidences)

        # I = I.clamp(0, 24)
        UV = np.clip(UV, a_min=0, a_max=1)*255
        if norm:
            I = I/24.
            UV = UV/255.
        iuv = np.concatenate([I[...,None],UV], axis=-1)
        iuv_list.append(iuv)
        # pdb.set_trace()
    iuv_batch = np.stack(iuv_list, axis=0)

    if not use_numpy:
        iuv_batch = torch.Tensor(iuv_batch).permute([0,3,1,2])

    return iuv_batch

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images 



## Ref: https://github.com/facebookresearch/DensePose/blob/master/notebooks/DensePose-RCNN-Texture-Transfer.ipynb
def TransferTexture(TextureIm,im,IUV,class_num=25):
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1,class_num):    ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        #####
        R = tex[:,:,0]
        G = tex[:,:,1]
        B = tex[:,:,2]
        ###############
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        ##  Get the RGB values from the texture images.
        R_im[IUV[:,:,0]==PartInd] = r_current_points
        G_im[IUV[:,:,0]==PartInd] = g_current_points
        B_im[IUV[:,:,0]==PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:,:,np.newaxis],G_im[:,:,np.newaxis],R_im[:,:,np.newaxis]), axis =2 ).astype(np.uint8)
    BG_MASK = generated_image==0
    generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image




