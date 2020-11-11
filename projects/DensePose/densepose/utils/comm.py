import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from detectron2.utils.comm import get_world_size


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device, norm=False):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    if norm:
        shift_x = shift_x.float()/(w * stride)
        shift_y = shift_y.float()/(h * stride)
        locations = torch.stack((shift_x, shift_y), dim=1)
    else:
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


# def meshgrid(self, height, width):
#     x_t = torch.matmul(
#         torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
#     y_t = torch.matmul(
#         torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

#     grid_x = x_t.view(1, height, width)
#     grid_y = y_t.view(1, height, width)
#     return grid_x, grid_y

# grid_x, grid_y = self.meshgrid(input_size[0], input_size[1])
# with torch.cuda.device(input.get_device()):
#     grid_x = torch.autograd.Variable(
#         grid_x.repeat([input.size()[0], 1, 1])).cuda()
#     grid_y = torch.autograd.Variable(
#         grid_y.repeat([input.size()[0], 1, 1])).cuda()

def compute_grid(h, w, device, norm=True):
    grid_x = torch.arange(
        0, w, step=1,
        dtype=torch.float32, device=device
    )
    grid_y = torch.arange(
        0, h, step=1,
        dtype=torch.float32, device=device
    )
    if norm:
        grid_y, grid_x = torch.meshgrid(grid_y/h, grid_x/w)
    else:
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack((grid_x, grid_y), dim=0)
    return grid


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
