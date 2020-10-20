# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from dataclasses import dataclass
from typing import Iterable, Optional
import torch, pdb
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Instances

from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType


def _linear_interpolation_utilities(v_norm, v0_src, size_src, v0_dst, size_dst, size_z):
    """
    Computes utility values for linear interpolation at points v.
    The points are given as normalized offsets in the source interval
    (v0_src, v0_src + size_src), more precisely:
        v = v0_src + v_norm * size_src / 256.0
    The computed utilities include lower points v_lo, upper points v_hi,
    interpolation weights v_w and flags j_valid indicating whether the
    points falls into the destination interval (v0_dst, v0_dst + size_dst).

    Args:
        v_norm (:obj: `torch.Tensor`): tensor of size N containing
            normalized point offsets
        v0_src (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of source intervals for normalized points
        size_src (:obj: `torch.Tensor`): tensor of size N containing
            source interval sizes for normalized points
        v0_dst (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of destination intervals
        size_dst (:obj: `torch.Tensor`): tensor of size N containing
            destination interval sizes
        size_z (int): interval size for data to be interpolated

    Returns:
        v_lo (:obj: `torch.Tensor`): int tensor of size N containing
            indices of lower values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_hi (:obj: `torch.Tensor`): int tensor of size N containing
            indices of upper values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_w (:obj: `torch.Tensor`): float tensor of size N containing
            interpolation weights
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size N containing
            0 for points outside the estimation interval
            (v0_est, v0_est + size_est) and 1 otherwise
    """
    v = v0_src + v_norm * size_src / 256.0
    j_valid = (v - v0_dst >= 0) * (v - v0_dst < size_dst)
    v_grid = (v - v0_dst) * size_z / size_dst
    v_lo = v_grid.floor().long().clamp(min=0, max=size_z - 1)
    v_hi = (v_lo + 1).clamp(max=size_z - 1)
    v_grid = torch.min(v_hi.float(), v_grid)
    v_w = v_grid - v_lo.float()
    return v_lo, v_hi, v_w, j_valid


class SingleTensorsHelper:
    """
    Convert gt data of batch images (each may contain multiple instances) 
    in to one tensor. Thus, the loss is only calculated once per batch.
    """
    def __init__(self, proposals_with_gt):

        with torch.no_grad():
            (
                index_uv_img,
                i_with_dp,
                bbox_xywh_est,
                bbox_xywh_gt,
                index_gt_all,
                x_norm,
                y_norm,
                u_gt_all,
                v_gt_all,
                s_gt,
                index_bbox,
                index_img_per_ins,
                i_height,
                i_width,
                gt_pointnum_per_ins,
            ) = _extract_single_tensors_from_matches(proposals_with_gt)

        for k, v in locals().items():
            if k not in ["self", "proposals_with_gt"]:
                setattr(self, k, v)


class BilinearInterpolationHelper:
    """
    Args:
        tensors_helper (SingleTensorsHelper)
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size M containing
            0 for points to be discarded and 1 for points to be selected
        y_lo (:obj: `torch.Tensor`): int tensor of indices of upper values
            in z_est for each point
        y_hi (:obj: `torch.Tensor`): int tensor of indices of lower values
            in z_est for each point
        x_lo (:obj: `torch.Tensor`): int tensor of indices of left values
            in z_est for each point
        x_hi (:obj: `torch.Tensor`): int tensor of indices of right values
            in z_est for each point
        w_ylo_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-left value weight for each point
        w_ylo_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-right value weight for each point
        w_yhi_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-left value weight for each point
        w_yhi_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-right value weight for each point
    """

    def __init__(
        self,
        tensors_helper,
        j_valid,
        y_lo,
        y_hi,
        x_lo,
        x_hi,
        w_ylo_xlo,
        w_ylo_xhi,
        w_yhi_xlo,
        w_yhi_xhi,
    ):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)

    @staticmethod
    def from_matches(tensors_helper, densepose_outputs_size):
        """
        Define biliner sampling location and weights according to tensors_helper.
        """

        zh, zw = densepose_outputs_size[2], densepose_outputs_size[3]

        x0_gt, y0_gt, w_gt, h_gt = tensors_helper.bbox_xywh_gt[tensors_helper.index_bbox].unbind(1)
        x0_est, y0_est, w_est, h_est = tensors_helper.bbox_xywh_est[
            tensors_helper.index_bbox
        ].unbind(dim=1)
        # pdb.set_trace()
        x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
            tensors_helper.x_norm, x0_gt, w_gt, x0_est, w_est, zw
        )
        y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
            tensors_helper.y_norm, y0_gt, h_gt, y0_est, h_est, zh
        )
        j_valid = jx_valid * jy_valid

        w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
        w_ylo_xhi = x_w * (1.0 - y_w)
        w_yhi_xlo = (1.0 - x_w) * y_w
        w_yhi_xhi = x_w * y_w

        return BilinearInterpolationHelper(
            tensors_helper,
            j_valid,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )

    def extract_at_points(
        self,
        z_est,
        slice_index_uv=None,
        w_ylo_xlo=None,
        w_ylo_xhi=None,
        w_yhi_xlo=None,
        w_yhi_xhi=None,
    ):
        """
        Extract ground truth values z_gt for valid point indices and estimated
        values z_est using bilinear interpolation over top-left (y_lo, x_lo),
        top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
        (y_hi, x_hi) values in z_est with corresponding weights:
        w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
        Use slice_index_uv to slice dim=1 in z_est
        """

        index_gt_all = self.tensors_helper.index_gt_all
        slice_index_uv = index_gt_all if slice_index_uv is None else slice_index_uv
        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        index_bbox = self.tensors_helper.index_bbox
        # pdb.set_trace()

        z_est_sampled = (
            z_est[index_bbox, slice_index_uv, self.y_lo, self.x_lo] * w_ylo_xlo
            + z_est[index_bbox, slice_index_uv, self.y_lo, self.x_hi] * w_ylo_xhi
            + z_est[index_bbox, slice_index_uv, self.y_hi, self.x_lo] * w_yhi_xlo
            + z_est[index_bbox, slice_index_uv, self.y_hi, self.x_hi] * w_yhi_xhi
        )
        return z_est_sampled


    @staticmethod
    def from_matches_diffHW(tensors_helper):
        """
        Define biliner sampling location and weights according to tensors_helper.
        """

        # zh, zw = densepose_outputs_size[2], densepose_outputs_size[3]

        x0_gt, y0_gt, w_gt, h_gt = tensors_helper.bbox_xywh_gt[tensors_helper.index_bbox].unbind(1)
        x0_est, y0_est, w_est, h_est = tensors_helper.bbox_xywh_est[
            tensors_helper.index_bbox
        ].unbind(dim=1)

        x_lo_list, x_hi_list, x_w_list = [], [], []
        y_lo_list, y_hi_list, y_w_list = [], [], []
        j_valid_list, w_ylo_xlo_list, w_ylo_xhi_list, w_yhi_xlo_list, w_yhi_xhi_list = [], [], [], [], []

        index_bbox = tensors_helper.index_bbox

        start = 0
        for idx, pointnum in enumerate(tensors_helper.gt_pointnum_per_ins):
            end = start + pointnum
            img_idx = tensors_helper.index_img_per_ins[idx]
            zh, zw = h_gt[start], w_gt[start].round()
            zh, zw = zh.round(), zw.round()
            # print(start,end,zh,zw)
            # pdb.set_trace()
            # tensors_helper.i_height[img_idx], tensors_helper.i_width[img_idx]
            x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
                tensors_helper.x_norm[start:end], x0_gt[start:end], w_gt[start:end], x0_est[start:end], w_est[start:end], zw
            )
            y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
                tensors_helper.y_norm[start:end], y0_gt[start:end], h_gt[start:end], y0_est[start:end], h_est[start:end], zh
            )
            j_valid = jx_valid * jy_valid
            # pdb.set_trace()

            w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
            w_ylo_xhi = x_w * (1.0 - y_w)
            w_yhi_xlo = (1.0 - x_w) * y_w
            w_yhi_xhi = x_w * y_w

            x_lo_list.append(x_lo)
            x_hi_list.append(x_hi)
            x_w_list.append(x_w)
            y_lo_list.append(y_lo)
            y_hi_list.append(y_hi)
            y_w_list.append(y_w)
            j_valid_list.append(j_valid)
            w_ylo_xlo_list.append(w_ylo_xlo)
            w_ylo_xhi_list.append(w_ylo_xhi)
            w_yhi_xlo_list.append(w_yhi_xlo)
            w_yhi_xhi_list.append(w_yhi_xhi)

            start = end

        j_valid = torch.cat(j_valid_list)
        y_lo = torch.cat(y_lo_list)
        y_hi = torch.cat(y_hi_list)
        x_lo = torch.cat(x_lo_list)
        x_hi = torch.cat(x_hi_list)
        w_ylo_xlo = torch.cat(w_ylo_xlo_list)
        w_ylo_xhi = torch.cat(w_ylo_xhi_list)
        w_yhi_xlo = torch.cat(w_yhi_xlo_list)
        w_yhi_xhi = torch.cat(w_yhi_xhi_list)

        # pdb.set_trace()
        # aa= BilinearInterpolationHelper(tensors_helper,j_valid,y_lo,y_hi,x_lo,x_hi,w_ylo_xlo,w_ylo_xhi,w_yhi_xlo,w_yhi_xhi,)

        return BilinearInterpolationHelper(
            tensors_helper,
            j_valid,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )

    def extract_at_points_globalIUV_diffHW(
        self,
        z_est,
        slice_index_uv=None,
        w_ylo_xlo=None,
        w_ylo_xhi=None,
        w_yhi_xlo=None,
        w_yhi_xhi=None,
        mode='bilinear',
        prefix=None
    ):
        """
        Extract ground truth values z_gt for valid point indices and estimated
        values z_est using bilinear interpolation over top-left (y_lo, x_lo),
        top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
        (y_hi, x_hi) values in z_est with corresponding weights:
        w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
        Use slice_index_uv to slice dim=1 in z_est
        """
        index_gt_all = self.tensors_helper.index_gt_all
        slice_index_uv = index_gt_all if slice_index_uv is None else slice_index_uv
        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        assert (self.tensors_helper.bbox_xywh_gt - self.tensors_helper.bbox_xywh_est).mean()==0
        x0_gt, y0_gt, w_gt, h_gt = self.tensors_helper.bbox_xywh_gt[self.tensors_helper.index_bbox].unbind(1)
        index_bbox = self.tensors_helper.index_bbox
        z_est_sampled_all = []
        start = 0
        for idx, pointnum in enumerate(self.tensors_helper.gt_pointnum_per_ins):
            end = start + pointnum
            img_idx = self.tensors_helper.index_img_per_ins[idx]
            logitH, logitW = z_est.shape[-2], z_est.shape[-1]
            imgH, imgW = self.tensors_helper.i_height[img_idx], self.tensors_helper.i_width[img_idx]
            x,y,w,h = x0_gt[start], y0_gt[start], w_gt[start], h_gt[start]
            sample_size = 256
            bbox_xywh_out = self.tensors_helper.bbox_xywh_gt[self.tensors_helper.index_bbox][start:start+1]
            bbox_xywh_out = bbox_xywh_out/imgW*logitW
            bbox_xywh_in = torch.tensor([[0,0,h,w],]).float().to(z_est.device)
            z_est_ins = _resample_data_v2(z_est[img_idx:img_idx+1], 
                                       bbox_xywh_in=bbox_xywh_in, 
                                       bbox_xywh_out=bbox_xywh_out,  
                                       wout=w.round().int(), hout=h.round().int(), 
                                       mode=mode)
            # z_est_ins = z_est[:,:,y:y+h,x:x+w]
            if slice_index_uv==slice(None):
                z_est_sampled = (
                      z_est_ins[0, slice(None), self.y_lo[start:end], self.x_lo[start:end]] * w_ylo_xlo[start:end]
                    + z_est_ins[0, slice(None), self.y_lo[start:end], self.x_hi[start:end]] * w_ylo_xhi[start:end]
                    + z_est_ins[0, slice(None), self.y_hi[start:end], self.x_lo[start:end]] * w_yhi_xlo[start:end]
                    + z_est_ins[0, slice(None), self.y_hi[start:end], self.x_hi[start:end]] * w_yhi_xhi[start:end]
                )
            else:
                z_est_sampled = (
                      z_est_ins[0, slice_index_uv[start:end], self.y_lo[start:end], self.x_lo[start:end]] * w_ylo_xlo[start:end]
                    + z_est_ins[0, slice_index_uv[start:end], self.y_lo[start:end], self.x_hi[start:end]] * w_ylo_xhi[start:end]
                    + z_est_ins[0, slice_index_uv[start:end], self.y_hi[start:end], self.x_lo[start:end]] * w_yhi_xlo[start:end]
                    + z_est_ins[0, slice_index_uv[start:end], self.y_hi[start:end], self.x_hi[start:end]] * w_yhi_xhi[start:end]
                )
            z_est_sampled_all.append(z_est_sampled)

            start = end

        return torch.cat(z_est_sampled_all,dim=-1)

    def extract_at_points_separatedS(
        self,
        z_est,
        slice_index_uv=None,
        w_ylo_xlo=None,
        w_ylo_xhi=None,
        w_yhi_xlo=None,
        w_yhi_xhi=None,
        mode='bilinear',
        prefix=None
    ):
        """
        Extract ground truth values z_gt for valid point indices and estimated
        values z_est using bilinear interpolation over top-left (y_lo, x_lo),
        top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
        (y_hi, x_hi) values in z_est with corresponding weights:
        w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
        Use slice_index_uv to slice dim=1 in z_est
        """
        index_gt_all = self.tensors_helper.index_gt_all
        slice_index_uv = index_gt_all if slice_index_uv is None else slice_index_uv
        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        assert (self.tensors_helper.bbox_xywh_gt - self.tensors_helper.bbox_xywh_est).mean()==0
        x0_gt, y0_gt, w_gt, h_gt = self.tensors_helper.bbox_xywh_gt[self.tensors_helper.index_bbox].unbind(1)
        index_bbox = self.tensors_helper.index_bbox
        z_est_sampled_all = []
        start = 0
        for idx, pointnum in enumerate(self.tensors_helper.gt_pointnum_per_ins):
            end = start + pointnum
            img_idx = self.tensors_helper.index_img_per_ins[idx]
            logitH, logitW = z_est.shape[-2], z_est.shape[-1]
            imgH, imgW = self.tensors_helper.i_height[img_idx], self.tensors_helper.i_width[img_idx]
            x,y,w,h = x0_gt[start], y0_gt[start], w_gt[start], h_gt[start]
            sample_size = 256
            bbox_xywh_out = self.tensors_helper.bbox_xywh_gt[self.tensors_helper.index_bbox][start:start+1]
            bbox_xywh_out = bbox_xywh_out/imgW*logitW
            bbox_xywh_in = torch.tensor([[0,0,h,w],]).float().to(z_est.device)
            # if z_est[idx:idx+1].shape[0]==0:
            #     pdb.set_trace()

            z_est_ins = _resample_data_v2(z_est[idx:idx+1], 
                                       bbox_xywh_in=bbox_xywh_in, 
                                       bbox_xywh_out=bbox_xywh_out,  
                                       wout=sample_size, hout=sample_size, 
                                       mode=mode)
            z_est_sampled = z_est_ins
            z_est_sampled_all.append(z_est_sampled)

            start = end

        return torch.cat(z_est_sampled_all,dim=0)

def _crop_resize(z_est, tensors_helper, crop_resize_size, mode):
    index_img = tensors_helper.index_img_per_ins.int()
    # z_resize = 
    z_resize = []
    for idx_img in range(z_est.shape[0]):
        z = F.interpolate(z_est[idx_img:idx_img+1], size=(tensors_helper.i_height[idx_img],tensors_helper.i_width[idx_img]), 
                            mode=mode, align_corners=True)
        z_resize.append(z)
    z_est_crop_resize = []
    for idx,idx_img in enumerate(index_img):
    # for idx in range(tensors_helper.bbox_xywh_gt.shape[0]):
        # pdb.set_trace()
        x,y,w,h = tensors_helper.bbox_xywh_gt[idx_img].int()
        # idx_img = index_img[idx].int()
        z = F.interpolate(z_resize[idx_img][:,:,y:y+h,x:x+w], size=(crop_resize_size,crop_resize_size), mode=mode, align_corners=(mode!='nearest'))
        z_est_crop_resize.append(z)
    z_est = torch.cat(z_est_crop_resize, dim=0)
    return z_est

def _resample_data(
    z, bbox_xywh_src, bbox_xywh_dst, wout, hout, mode="nearest", padding_mode="zeros"
):
    """
    Note: bbox_xywh_src is the final output bbox, bbox_xywh_dst is where to sample the
    pixels, i.e. flow's destination.
    """
    """
    Args:
        z (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with data to be
            resampled
        bbox_xywh_src (:obj: `torch.Tensor`): tensor of size (N,4) containing
            source bounding boxes in format XYWH
        bbox_xywh_dst (:obj: `torch.Tensor`): tensor of size (N,4) containing
            destination bounding boxes in format XYWH
    Return:
        zresampled (:obj: `torch.Tensor`): tensor of size (N, C, Hout, Wout)
            with resampled values of z, where D is the discretization size
    """
    n = bbox_xywh_src.size(0)
    assert n == bbox_xywh_dst.size(0), (
        "The number of "
        "source ROIs for resampling ({}) should be equal to the number "
        "of destination ROIs ({})".format(bbox_xywh_src.size(0), bbox_xywh_dst.size(0))
    )
    x0src, y0src, wsrc, hsrc = bbox_xywh_src.unbind(dim=1)
    x0dst, y0dst, wdst, hdst = bbox_xywh_dst.unbind(dim=1)
    x0dst_norm = 2 * (x0dst - x0src) / wsrc - 1
    y0dst_norm = 2 * (y0dst - y0src) / hsrc - 1
    x1dst_norm = 2 * (x0dst + wdst - x0src) / wsrc - 1
    y1dst_norm = 2 * (y0dst + hdst - y0src) / hsrc - 1
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w_expanded = grid_w[None, None, :].expand(n, hout, wout)
    grid_h_expanded = grid_h[None, :, None].expand(n, hout, wout)
    dx_expanded = (x1dst_norm - x0dst_norm)[:, None, None].expand(n, hout, wout)
    dy_expanded = (y1dst_norm - y0dst_norm)[:, None, None].expand(n, hout, wout)
    x0_expanded = x0dst_norm[:, None, None].expand(n, hout, wout)
    y0_expanded = y0dst_norm[:, None, None].expand(n, hout, wout)
    grid_x = grid_w_expanded * dx_expanded + x0_expanded
    grid_y = grid_h_expanded * dy_expanded + y0_expanded
    grid = torch.stack((grid_x, grid_y), dim=3)
    # pdb.set_trace()
    # resample Z from (N, C, H, W) into (N, C, Hout, Wout)
    zresampled = F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return zresampled

def _resample_data_v2(
    z, bbox_xywh_in, bbox_xywh_out, wout, hout, mode="nearest", padding_mode="zeros"
):
    """
    Note: bbox_xywh_in is the final output bbox, bbox_xywh_out is where to sample the
    pixels, i.e. flow's destination.
    """
    """
    Args:
        z (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with data to be
            resampled
        bbox_xywh_in (:obj: `torch.Tensor`): tensor of size (N,4) containing
            source bounding boxes in format XYWH
        bbox_xywh_out (:obj: `torch.Tensor`): tensor of size (N,4) containing
            destination bounding boxes in format XYWH
    Return:
        zresampled (:obj: `torch.Tensor`): tensor of size (N, C, Hout, Wout)
            with resampled values of z, where D is the discretization size
    """
    n = bbox_xywh_in.size(0)
    assert n == bbox_xywh_out.size(0), (
        "The number of "
        "source ROIs for resampling ({}) should be equal to the number "
        "of destination ROIs ({})".format(bbox_xywh_in.size(0), bbox_xywh_out.size(0))
    )
    x0src, y0src, wsrc, hsrc = bbox_xywh_in.unbind(dim=1)
    x0dst, y0dst, wdst, hdst = bbox_xywh_out.unbind(dim=1)
    z_h, z_w = z.shape[-2:]
    x0dst_norm = 2 * (x0dst - x0src) / z_w - 1
    y0dst_norm = 2 * (y0dst - y0src) / z_h - 1
    x1dst_norm = 2 * (x0dst + wdst - x0src) / z_w - 1
    y1dst_norm = 2 * (y0dst + hdst - y0src) / z_h - 1
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w_expanded = grid_w[None, None, :].expand(n, hout, wout)
    grid_h_expanded = grid_h[None, :, None].expand(n, hout, wout)
    dx_expanded = (x1dst_norm - x0dst_norm)[:, None, None].expand(n, hout, wout)
    dy_expanded = (y1dst_norm - y0dst_norm)[:, None, None].expand(n, hout, wout)
    x0_expanded = x0dst_norm[:, None, None].expand(n, hout, wout)
    y0_expanded = y0dst_norm[:, None, None].expand(n, hout, wout)

    # dx_expanded = wdst

    grid_x = grid_w_expanded * dx_expanded + x0_expanded
    grid_y = grid_h_expanded * dy_expanded + y0_expanded
    grid = torch.stack((grid_x, grid_y), dim=3)
    # resample Z from (N, C, H, W) into (N, C, Hout, Wout)
    zresampled = F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return zresampled

def _extract_single_tensors_from_matches_one_image(
    proposals_targets, bbox_with_dp_offset, bbox_global_offset
):
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    # Ibbox_all == k should be true for all data that corresponds
    # to bbox_xywh_gt[k] and bbox_xywh_est[k]
    # index k here is global wrt images
    i_bbox_all = []
    # at offset k (k is global) contains index of bounding box data
    # within densepose output tensor
    i_with_dp = []

    boxes_xywh_est = proposals_targets.proposal_boxes.clone()
    boxes_xywh_gt = proposals_targets.gt_boxes.clone()
    n_i = len(boxes_xywh_est)
    assert n_i == len(boxes_xywh_gt)

    if n_i:
        boxes_xywh_est.tensor[:, 2] -= boxes_xywh_est.tensor[:, 0]
        boxes_xywh_est.tensor[:, 3] -= boxes_xywh_est.tensor[:, 1]
        boxes_xywh_gt.tensor[:, 2] -= boxes_xywh_gt.tensor[:, 0]
        boxes_xywh_gt.tensor[:, 3] -= boxes_xywh_gt.tensor[:, 1]
        # pdb.set_trace()
        if hasattr(proposals_targets, "gt_densepose"):
            densepose_gt = proposals_targets.gt_densepose
            for k, box_xywh_est, box_xywh_gt, dp_gt in zip(
                range(n_i), boxes_xywh_est.tensor, boxes_xywh_gt.tensor, densepose_gt
            ):
                if (dp_gt is not None) and (len(dp_gt.x) > 0):
                    i_gt_all.append(dp_gt.i)
                    x_norm_all.append(dp_gt.x)
                    y_norm_all.append(dp_gt.y)
                    u_gt_all.append(dp_gt.u)
                    v_gt_all.append(dp_gt.v)
                    s_gt_all.append(dp_gt.segm.unsqueeze(0))
                    bbox_xywh_gt_all.append(box_xywh_gt.view(-1, 4))
                    bbox_xywh_est_all.append(box_xywh_est.view(-1, 4))
                    i_bbox_k = torch.full_like(dp_gt.i, bbox_with_dp_offset + len(i_with_dp))
                    i_bbox_all.append(i_bbox_k)
                    i_with_dp.append(bbox_global_offset + k)


    return (
        i_gt_all,
        x_norm_all,
        y_norm_all,
        u_gt_all,
        v_gt_all,
        s_gt_all,
        bbox_xywh_gt_all,
        bbox_xywh_est_all,
        i_bbox_all,
        i_with_dp,
    )

def _extract_single_tensors_from_matches(proposals_with_targets):
    i_img = []
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    i_bbox_all = []
    i_with_dp_all = []
    i_img_per_ins_all = []
    img_height_all = []
    img_width_all = []
    gt_pointnum_per_ins_all = []
    n = 0
    for i, proposals_targets_per_image in enumerate(proposals_with_targets):
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        (
            i_gt_img,
            x_norm_img,
            y_norm_img,
            u_gt_img,
            v_gt_img,
            s_gt_img,
            bbox_xywh_gt_img,
            bbox_xywh_est_img,
            i_bbox_img,
            i_with_dp_img,
        ) = _extract_single_tensors_from_matches_one_image(  # noqa
            proposals_targets_per_image, len(i_with_dp_all), n
        )
        i_gt_all.extend(i_gt_img)
        x_norm_all.extend(x_norm_img)
        y_norm_all.extend(y_norm_img)
        u_gt_all.extend(u_gt_img)
        v_gt_all.extend(v_gt_img)
        s_gt_all.extend(s_gt_img)
        bbox_xywh_gt_all.extend(bbox_xywh_gt_img)
        bbox_xywh_est_all.extend(bbox_xywh_est_img)
        i_bbox_all.extend(i_bbox_img)
        i_with_dp_all.extend(i_with_dp_img)
        i_img.extend([i] * len(i_with_dp_img))
        i_img_per_ins_all.extend([torch.ones_like(bb)*i for bb in i_bbox_img])
        H, W = proposals_targets_per_image._image_size
        # pdb.set_trace()
        # img_height_all.extend([H]*len(i_bbox_img))
        # img_width_all.extend([W]*len(i_bbox_img))
        img_height_all.extend([H])
        img_width_all.extend([W])
        gt_pointnum_per_ins_all.extend([u.shape[0] for u in u_gt_img])
        # gt_pointnum_per_ins_all.append(u_gt_img[0].shape[0])
        n += n_i
    # concatenate all data into a single tensor
    if (n > 0) and (len(i_with_dp_all) > 0):
        i_gt = torch.cat(i_gt_all, 0).long()
        x_norm = torch.cat(x_norm_all, 0)
        y_norm = torch.cat(y_norm_all, 0)
        u_gt = torch.cat(u_gt_all, 0)
        v_gt = torch.cat(v_gt_all, 0)
        s_gt = torch.cat(s_gt_all, 0)
        bbox_xywh_gt = torch.cat(bbox_xywh_gt_all, 0)
        bbox_xywh_est = torch.cat(bbox_xywh_est_all, 0)
        i_bbox = torch.cat(i_bbox_all, 0).long()
        i_img_per_ins = torch.cat(i_img_per_ins_all, 0).long()
        i_height = img_height_all #torch.stack(img_height_all)
        i_width = img_width_all #torch.stack(img_width_all, 0)
        gt_pointnum_per_ins = gt_pointnum_per_ins_all
        # pdb.set_trace()
    else:
        i_gt = None
        x_norm = None
        y_norm = None
        u_gt = None
        v_gt = None
        s_gt = None
        bbox_xywh_gt = None
        bbox_xywh_est = None
        i_bbox = None
        i_img_per_ins = None
        i_height = None
        i_width = None
        gt_pointnum_per_ins = None
    # print(i_bbox)
    # pdb.set_trace()
    return (
        i_img,
        i_with_dp_all,
        bbox_xywh_est,
        bbox_xywh_gt,
        i_gt,
        x_norm,
        y_norm,
        u_gt,
        v_gt,
        s_gt,
        i_bbox,
        i_img_per_ins,
        i_height,
        i_width,
        gt_pointnum_per_ins,
    )


@dataclass
class DataForMaskLoss:
    """
    Contains mask GT and estimated data for proposals from multiple images:
    """

    # tensor of size (K, H, W) containing GT labels
    masks_gt: Optional[torch.Tensor] = None
    # tensor of size (K, C, H, W) containing estimated scores
    masks_est: Optional[torch.Tensor] = None


def _extract_data_for_mask_loss_from_matches(
    proposals_targets: Iterable[Instances], estimated_segm: torch.Tensor
) -> DataForMaskLoss:
    """
    Extract data for mask loss from instances that contain matched GT and
    estimated bounding boxes.
    Args:
        proposals_targets: Iterable[Instances]
            matched GT and estimated results, each item in the iterable
            corresponds to data in 1 image
        estimated_segm: torch.Tensor if size
            size to which GT masks are resized
    Return:
        masks_est: tensor(K, C, H, W) of float - class scores
        masks_gt: tensor(K, H, W) of int64 - labels
    """
    data = DataForMaskLoss()
    masks_gt = []
    offset = 0
    assert estimated_segm.shape[2] == estimated_segm.shape[3], (
        f"Expected estimated segmentation to have a square shape, "
        f"but the actual shape is {estimated_segm.shape[2:]}"
    )
    mask_size = estimated_segm.shape[2]
    num_proposals = sum(inst.proposal_boxes.tensor.size(0) for inst in proposals_targets)
    num_estimated = estimated_segm.shape[0]
    assert (
        num_proposals == num_estimated
    ), "The number of proposals {} must be equal to the number of estimates {}".format(
        num_proposals, num_estimated
    )

    for proposals_targets_per_image in proposals_targets:
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        gt_masks_per_image = proposals_targets_per_image.gt_masks.crop_and_resize(
            proposals_targets_per_image.proposal_boxes.tensor, mask_size
        ).to(device=estimated_segm.device)
        masks_gt.append(gt_masks_per_image)
        offset += n_i
    if masks_gt:
        data.masks_est = estimated_segm
        data.masks_gt = torch.cat(masks_gt, dim=0)
    return data


class IIDIsotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of iid residuals with isotropic covariance:
    $Sigma_i = sigma_i^2 I$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi) + 2 log sigma_i^2 + ||delta_i||^2 / sigma_i^2)$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: float):
        super(IIDIsotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sigma_u: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ):
        # compute $\sigma_i^2$
        # use sigma_lower_bound to avoid degenerate solution for variance
        # (sigma -> 0)
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        # compute \|delta_i\|^2
        delta_t_delta = (u - target_u) ** 2 + (v - target_v) ** 2
        # the total loss from the formula above:
        loss = 0.5 * (self.log2pi + 2 * torch.log(sigma2) + delta_t_delta / sigma2)
        return loss.sum()


class IndepAnisotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of independent residuals with anisotropic covariances:
    $Sigma_i = sigma_i^2 I + r_i r_i^T$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi)
      + log sigma_i^2 (sigma_i^2 + ||r_i||^2)
      + ||delta_i||^2 / sigma_i^2
      - <delta_i, r_i>^2 / (sigma_i^2 * (sigma_i^2 + ||r_i||^2)))$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: float):
        super(IndepAnisotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sigma_u: torch.Tensor,
        kappa_u_est: torch.Tensor,
        kappa_v_est: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ):
        # compute $\sigma_i^2$
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        # compute \|r_i\|^2
        r_sqnorm2 = kappa_u_est ** 2 + kappa_v_est ** 2
        delta_u = u - target_u
        delta_v = v - target_v
        # compute \|delta_i\|^2
        delta_sqnorm = delta_u ** 2 + delta_v ** 2
        delta_u_r_u = delta_u * kappa_u_est
        delta_v_r_v = delta_v * kappa_v_est
        # compute the scalar product <delta_i, r_i>
        delta_r = delta_u_r_u + delta_v_r_v
        # compute squared scalar product <delta_i, r_i>^2
        delta_r_sqnorm = delta_r ** 2
        denom2 = sigma2 * (sigma2 + r_sqnorm2)
        loss = 0.5 * (
            self.log2pi + torch.log(denom2) + delta_sqnorm / sigma2 - delta_r_sqnorm / denom2
        )
        return loss.sum()


class DensePoseLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
            self.uv_loss_with_confidences = IIDIsotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )
        elif self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.INDEP_ANISO:
            self.uv_loss_with_confidences = IndepAnisotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )
        self.densepose_size = 256

    def __call__(self, proposals_with_gt, densepose_outputs, densepose_confidences, bbox_free=False):
        if not self.segm_trained_by_masks:
            if bbox_free:
                return self.produce_densepose_bbox_free_losses(
                    proposals_with_gt, densepose_outputs, densepose_confidences
                )
            else:
                return self.produce_densepose_losses(
                    proposals_with_gt, densepose_outputs, densepose_confidences
                )
        else:
            losses = {}
            if bbox_free:
                losses_densepose = self.produce_densepose_bbox_free_losses(
                    proposals_with_gt, densepose_outputs, densepose_confidences
                )
                losses.update(losses_densepose)
            else:
                losses_densepose = self.produce_densepose_losses(
                    proposals_with_gt, densepose_outputs, densepose_confidences
                )
                losses.update(losses_densepose)
                losses_mask = self.produce_mask_losses(
                    proposals_with_gt, densepose_outputs, densepose_confidences
                )
                losses.update(losses_mask)
            return losses

    def produce_fake_mask_losses(self, densepose_outputs):
        losses = {}
        segm_scores, _, _, _ = densepose_outputs
        losses["loss_densepose_S"] = segm_scores.sum() * 0
        return losses

    def produce_mask_losses(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        if not len(proposals_with_gt):
            return self.produce_fake_mask_losses(densepose_outputs)
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        segm_scores, _, _, _ = densepose_outputs
        with torch.no_grad():
            mask_loss_data = _extract_data_for_mask_loss_from_matches(
                proposals_with_gt, segm_scores
            )
        if (mask_loss_data.masks_gt is None) or (mask_loss_data.masks_est is None):
            return self.produce_fake_mask_losses(densepose_outputs)
        losses["loss_densepose_S"] = (
            F.cross_entropy(mask_loss_data.masks_est, mask_loss_data.masks_gt.long()) * self.w_segm
        )
        return losses

    def produce_fake_densepose_losses(self, densepose_outputs, densepose_confidences):
        # we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) losses in the form Tensor.sum() * 0
        s, index_uv, u, v = densepose_outputs
        conf_type = self.confidence_model_cfg.uv_confidence.type
        (
            sigma_1,
            sigma_2,
            kappa_u,
            kappa_v,
            fine_segm_confidence,
            coarse_segm_confidence,
        ) = densepose_confidences
        losses = {}
        losses["loss_densepose_I"] = index_uv.sum() * 0
        if not self.segm_trained_by_masks:
            losses["loss_densepose_S"] = s.sum() * 0
        if self.confidence_model_cfg.uv_confidence.enabled:
            losses["loss_densepose_UV"] = (u.sum() + v.sum()) * 0
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                losses["loss_densepose_UV"] += sigma_2.sum() * 0
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                losses["loss_densepose_UV"] += (sigma_2.sum() + kappa_u.sum() + kappa_v.sum()) * 0
        else:
            losses["loss_densepose_U"] = u.sum() * 0
            losses["loss_densepose_V"] = v.sum() * 0
        return losses

    def produce_densepose_bbox_free_losses(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        """
        Calculate IUV loss for whole image instead for one-by-one person bbox;
        # Segm loss will not be calculated, as here is only for global whole image losses.
        # Segm loss is calculated in dynamic mask head w/wo instance mask loss.
        """
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        s, index_uv, u, v = densepose_outputs
        assert u.size(2) == v.size(2)
        assert u.size(3) == v.size(3)
        assert u.size(2) == index_uv.size(2)
        assert u.size(3) == index_uv.size(3)
        densepose_outputs_size = u.size()
        
        # pdb.set_trace()
        # import imageio
        # imageio.imwrite('tmp/tmp_index_uv_-1.png', index_uv[0,-1].detach().cpu().numpy())

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)
        (
            sigma_1,
            sigma_2,
            kappa_u,
            kappa_v,
            fine_segm_confidence,
            coarse_segm_confidence,
        ) = densepose_confidences
        conf_type = self.confidence_model_cfg.uv_confidence.type

        tensors_helper = SingleTensorsHelper(proposals_with_gt)
        n_batch = len(tensors_helper.i_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)

        densepose_outputs_size = (None,None,256,256) ## manually eset to densepose label size
        # interpolator = BilinearInterpolationHelper.from_matches(
        #     tensors_helper, densepose_outputs_size
        # )
        interpolator = BilinearInterpolationHelper.from_matches_diffHW(
            tensors_helper
        )

        j_valid_fg = interpolator.j_valid * (tensors_helper.index_gt_all > 0)

        u_gt = tensors_helper.u_gt_all[j_valid_fg]
        # print("a",u.shape)
        # print("b",tensors_helper.i_with_dp)
        # print(u[tensors_helper.i_with_dp])
        # u_est_all = interpolator.extract_at_points(u[tensors_helper.i_with_dp])

        # u = _crop_resize(u, tensors_helper, self.densepose_size, mode='bilinear')[tensors_helper.i_with_dp]
        # pdb.set_trace()
        # u = u[tensors_helper.i_with_dp]
        u_est_all = interpolator.extract_at_points_globalIUV_diffHW(u, prefix="u")
        u_est = u_est_all[j_valid_fg]

        v_gt = tensors_helper.v_gt_all[j_valid_fg]
        # v_est_all = interpolator.extract_at_points(v[tensors_helper.i_with_dp])
        # v = _crop_resize(v, tensors_helper, self.densepose_size, mode='bilinear')[tensors_helper.i_with_dp]
        # v_est_all = interpolator.extract_at_points(v)
        v_est_all = interpolator.extract_at_points_globalIUV_diffHW(v, prefix="v")
        v_est = v_est_all[j_valid_fg]

        index_uv_gt = tensors_helper.index_gt_all[interpolator.j_valid]
        # index_uv_est_all = interpolator.extract_at_points(
        #     index_uv[tensors_helper.i_with_dp],
        #     slice_index_uv=slice(None),
        #     w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
        #     w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
        #     w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
        #     w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        # )
        # index_uv = _crop_resize(index_uv, tensors_helper, self.densepose_size, mode='nearest')[tensors_helper.i_with_dp]
        # print("index_uv")
        # pdb.set_trace()
        index_uv_est_all = interpolator.extract_at_points_globalIUV_diffHW(
            index_uv,
            slice_index_uv=slice(None),
            mode='nearest',
            prefix="index_uv"
        ).permute([1,0])
        index_uv_est = index_uv_est_all[interpolator.j_valid, :]

        if self.confidence_model_cfg.uv_confidence.enabled:
            # sigma2 = _crop_resize(sigma2, tensors_helper, self.densepose_size, mode='bilinear')[tensors_helper.i_with_dp]
            # sigma_2_est_all = interpolator.extract_at_points(sigma2)
            sigma_2_est_all = interpolator.extract_at_points_globalIUV_diffHW(sigma2)
            sigma_2_est = sigma_2_est_all[j_valid_fg]
            if conf_type in [DensePoseUVConfidenceType.INDEP_ANISO]:
                # kappa_u = _crop_resize(kappa_u, tensors_helper, self.densepose_size, mode='bilinear')[tensors_helper.i_with_dp]
                kappa_u_est_all = interpolator.extract_at_points(kappa_u)
                kappa_u_est_all = interpolator.extract_at_points_globalIUV_diffHW(kappa_u)
                kappa_u_est = kappa_u_est_all[j_valid_fg]
                # kappa_v = _crop_resize(kappa_v, tensors_helper, self.densepose_size, mode='bilinear')[tensors_helper.i_with_dp]
                # kappa_v_est_all = interpolator.extract_at_points(kappa_v)
                kappa_v_est_all = interpolator.extract_at_points_globalIUV_diffHW(kappa_v)
                kappa_v_est = kappa_v_est_all[j_valid_fg]

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        if not self.segm_trained_by_masks and self.w_segm>0:
            # s_est = s[tensors_helper.i_with_dp]
            # s_est = _crop_resize(s, tensors_helper, self.densepose_size, mode='nearest')[tensors_helper.i_with_dp]
            
            s_gt = tensors_helper.s_gt
            s = s[tensors_helper.i_with_dp]
            s_est = interpolator.extract_at_points_separatedS(
                s,
                slice_index_uv=slice(None),
                mode='nearest',
                prefix="s"
            )#.permute([0,2,3,1])
            # print('==> s_gt.shape:',s_gt.shape, 's.shape:', s.shape, 's_est.shape:', s_est.shape)
            # pdb.set_trace()



            # with torch.no_grad():
            #     # if bbox_free:
            #     # s_gt = F.interpolate(tensors_helper.s_gt.unsqueeze(1), size=(s_est.shape[-2],s_est.shape[-1]), 
            #     #                 mode="nearest").squeeze(1)
            #     n,c,h,w = s_est.shape
            #     # pdb.set_trace()
            #     s_gt = _resample_data(
            #         tensors_helper.s_gt.unsqueeze(1),
            #         tensors_helper.bbox_xywh_gt,
            #         tensors_helper.bbox_xywh_est,
            #         self.densepose_size,
            #         self.densepose_size,
            #         mode="nearest",
            #         padding_mode="zeros",
            #     ).squeeze(1)

            #     # pdb.set_trace()
            #     # s_gt = _resample_data(tensors_helper.s_gt.unsqueeze(1),tensors_helper.bbox_xywh_gt,tensors_helper.bbox_xywh_est,
            #     #     self.heatmap_size,
            #     #     self.heatmap_size,
            #     #     mode="nearest",
            #     #     padding_mode="zeros",
            #     # ).squeeze(1)
            #     import imageio
            #     # imageio.imwrite('tmp/tmp_tensors_helper_s_gt.png', tensors_helper.s_gt[0].detach().cpu().numpy())
            #     # imageio.imwrite('tmp/tmp_s_gt.png', s_gt[0].detach().cpu().numpy())

            #     if proposals_with_gt[0].proposal_boxes!=proposals_with_gt[0].gt_boxes:
            #         pdb.set_trace()

            #     # else:
            #     #     s_gt = _resample_data(
            #     #         tensors_helper.s_gt.unsqueeze(1),
            #     #         tensors_helper.bbox_xywh_gt,
            #     #         tensors_helper.bbox_xywh_est,
            #     #         self.heatmap_size,
            #     #         self.heatmap_size,
            #     #         mode="nearest",
            #     #         padding_mode="zeros",
            #     #     ).squeeze(1)

        # add point-based losses:
        if self.confidence_model_cfg.uv_confidence.enabled:
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                uv_loss = (
                    self.uv_loss_with_confidences(u_est, v_est, sigma_2_est, u_gt, v_gt)
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                uv_loss = (
                    self.uv_loss_with_confidences(
                        u_est, v_est, sigma_2_est, kappa_u_est, kappa_v_est, u_gt, v_gt
                    )
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            else:
                raise ValueError(f"Unknown confidence model type: {conf_type}")
        else:
            u_loss = F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points
            losses["loss_densepose_U"] = u_loss
            v_loss = F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points
            losses["loss_densepose_V"] = v_loss
        index_uv_loss = F.cross_entropy(index_uv_est, index_uv_gt.long()) * self.w_part
        losses["loss_densepose_I"] = index_uv_loss

        if not self.segm_trained_by_masks and self.w_segm>0:
            if self.n_segm_chan == 2:
                s_gt = s_gt > 0
            s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
            losses["loss_densepose_S"] = s_loss

        return losses

    def produce_densepose_losses(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        s, index_uv, u, v = densepose_outputs
        assert u.size(2) == v.size(2)
        assert u.size(3) == v.size(3)
        assert u.size(2) == index_uv.size(2)
        assert u.size(3) == index_uv.size(3)
        densepose_outputs_size = u.size()

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)
        (
            sigma_1,
            sigma_2,
            kappa_u,
            kappa_v,
            fine_segm_confidence,
            coarse_segm_confidence,
        ) = densepose_confidences
        conf_type = self.confidence_model_cfg.uv_confidence.type

        tensors_helper = SingleTensorsHelper(proposals_with_gt)
        n_batch = len(tensors_helper.i_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)

        interpolator = BilinearInterpolationHelper.from_matches(
            tensors_helper, densepose_outputs_size
        )

        j_valid_fg = interpolator.j_valid * (tensors_helper.index_gt_all > 0)

        u_gt = tensors_helper.u_gt_all[j_valid_fg]
        u_est_all = interpolator.extract_at_points(u[tensors_helper.i_with_dp])
        u_est = u_est_all[j_valid_fg]
        # pdb.set_trace()

        v_gt = tensors_helper.v_gt_all[j_valid_fg]
        v_est_all = interpolator.extract_at_points(v[tensors_helper.i_with_dp])
        v_est = v_est_all[j_valid_fg]

        index_uv_gt = tensors_helper.index_gt_all[interpolator.j_valid]
        index_uv_est_all = interpolator.extract_at_points(
            index_uv[tensors_helper.i_with_dp],
            slice_index_uv=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )
        index_uv_est = index_uv_est_all[interpolator.j_valid, :]

        if self.confidence_model_cfg.uv_confidence.enabled:
            sigma_2_est_all = interpolator.extract_at_points(sigma_2[tensors_helper.i_with_dp])
            sigma_2_est = sigma_2_est_all[j_valid_fg]
            if conf_type in [DensePoseUVConfidenceType.INDEP_ANISO]:
                kappa_u_est_all = interpolator.extract_at_points(kappa_u[tensors_helper.i_with_dp])
                kappa_u_est = kappa_u_est_all[j_valid_fg]
                kappa_v_est_all = interpolator.extract_at_points(kappa_v[tensors_helper.i_with_dp])
                kappa_v_est = kappa_v_est_all[j_valid_fg]

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        if not self.segm_trained_by_masks:
            s_est = s[tensors_helper.i_with_dp]
            with torch.no_grad():
                s_gt = _resample_data(
                    tensors_helper.s_gt.unsqueeze(1),
                    tensors_helper.bbox_xywh_gt,
                    tensors_helper.bbox_xywh_est,
                    self.heatmap_size,
                    self.heatmap_size,
                    mode="nearest",
                    padding_mode="zeros",
                ).squeeze(1)

        # add point-based losses:
        if self.confidence_model_cfg.uv_confidence.enabled:
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                uv_loss = (
                    self.uv_loss_with_confidences(u_est, v_est, sigma_2_est, u_gt, v_gt)
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                uv_loss = (
                    self.uv_loss_with_confidences(
                        u_est, v_est, sigma_2_est, kappa_u_est, kappa_v_est, u_gt, v_gt
                    )
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            else:
                raise ValueError(f"Unknown confidence model type: {conf_type}")
        else:
            u_loss = F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points
            losses["loss_densepose_U"] = u_loss
            v_loss = F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points
            losses["loss_densepose_V"] = v_loss
        index_uv_loss = F.cross_entropy(index_uv_est, index_uv_gt.long()) * self.w_part
        losses["loss_densepose_I"] = index_uv_loss
        # if index_uv_gt.min()==0:

        if not self.segm_trained_by_masks:
            if self.n_segm_chan == 2:
                s_gt = s_gt > 0
            s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
            losses["loss_densepose_S"] = s_loss
        return losses
