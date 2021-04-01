import imageio, os, sys, tqdm, pdb
import numpy as np
import torch
import torch.nn.functional as F


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

class flow_model_wrapper:
    def __init__(self, device):
    # def create_and_load_netFlow(self, device):
        # parser.add_argument('--model', help="restore checkpoint")
        # parser.add_argument('--seq_img_dir', help="sequence images for evaluation")
        # parser.add_argument('--backward_flow', action='store_true', help='calculate flow from i+1 to i')
        # parser.add_argument('--small', action='store_true', help='use small model')
        # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        # args = parser.parse_args()
        sys.path.append('./RAFT/core')
        from raft import RAFT
        class Args:
            def __init__(self):
                # self.model = "./RAFT/models/raft-sintel.pth"
                # self.small = False
                # self.mixed_precision = False
                # self.dropout = 0
                # self.alternate_corr = False
                self.model = "./RAFT/models/raft-small.pth"
                self.small = True
                self.mixed_precision = False
                self.dropout = 0
                self.alternate_corr = False
        args = Args()

        # model = RAFT(args)
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model, map_location=device))
        self.model.eval()
        print("Create and load model successfully")
        # return model

    def pred_flow(self, img0, img1, iters=20):
        try:
            assert img0.min()>=0 and img0.max()>=10 and img0.max()<=255, "input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max())
        except:
            print("input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max()))
            raise ValueError

        padder = InputPadder(img0.shape, mode='sintel')
        img0, img1 = padder.pad(img0, img1)

        flow_low, flow_pr = self.model(img0, img1, iters, test_mode=True)
        flow = padder.unpad(flow_pr)
        return flow

    def tensor_warp_via_flow(self, tensor, flow):
        b, _, h, w = tensor.shape
        coords = self.flow2coord(flow).permute([0,2,3,1]) # [0,h-1], [0,w-1]
        tensor = F.grid_sample(tensor, coords) #, mode='bilinear', align_corners=True)
        return tensor

    def flow2coord(self, flow):
        def meshgrid(height, width):
            x_t = torch.matmul(
                torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
            y_t = torch.matmul(
                torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

            grid_x = x_t.view(1, 1, height, width)
            grid_y = y_t.view(1, 1, height, width)
            return grid_x, grid_y
            # return torch.cat([grid_x,grid_y], dim=-1)

        b, _, h, w = flow.shape
        grid_x, grid_y = meshgrid(h, w)
        coord_x = flow[:,0:1]/w + grid_x.to(flow.device)
        coord_y = flow[:,1:2]/h + grid_y.to(flow.device)
        return torch.cat([coord_x,coord_y], dim=1)


data_root = "/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/Pose/youtube2"
vid_infos = {"02_clip02":0, 
			"04_clip02":0, 
			"06_clip02":10, 
			"07_clip02":0, 
			"08_clip02":5, 
			"09_clip02":0, 
			"11_clip02":0, 
			"15_clip02":0, 
			"17_clip02":0, 
			"18_clip02":0, 
			"19_clip02":6}
# vid_infos = {"05_clip02":0,}
# methods = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_smooth2_smoothCoarseSegm",
# 		   "densepose_rcnn_R_50_FPN_DL_s1x"]
# methods = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3",
# 			"densepose_rcnn_R_50_FPN_DL_s1x"
# 			]
methods = ["densepose_rcnn_R_50_FPN_DL_s1x"
			]
short_side_size = 512
aa=torch.tensor([0]).cuda()
device=aa.device
flow_model = flow_model_wrapper(device)


for method in methods:
	mse_list = []
	for vid_name in tqdm.tqdm(sorted(list(vid_infos.keys()))):
		# w = imageio.get_writer(os.path.join(data_root, vid_name, "out_vid_{}.mp4".format(vid_name)), fps=24)

		# if int(vid_name.split("_")[0])%2==0:
		# 	idx = 0
		# else:
		# 	idx = 1

		start_sec = vid_infos[vid_name]
		for i in range(start_sec*24+1, (start_sec+20)*24+1):
		# for i in range(start_sec*1+1, (start_sec+1)*1+1):
			# print(i)
			if i>1:
				img_tgt = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i)))
				img_ref = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i-1)))
				img_tgt = torch.tensor(img_tgt).cuda().float().permute([2,0,1])[None,...]
				img_ref = torch.tensor(img_ref).cuda().float().permute([2,0,1])[None,...]
				h, w = img_tgt.shape[-2:]
				if h>w:
					w_tgt = short_side_size
					h_tgt = int(w_tgt/w*h)
				else:
					h_tgt = short_side_size
					w_tgt = int(h_tgt/h*w)
				# print(h_tgt,w_tgt)
				# if w_tgt>1000:
				# 	pdb.set_trace()
				img_tgt = F.interpolate(img_tgt, [h_tgt,w_tgt], mode="bilinear")
				img_ref = F.interpolate(img_ref, [h_tgt,w_tgt], mode="bilinear")
				# pdb.set_trace()
				rgb_flow_fw = flow_model.pred_flow(img_tgt, img_ref)
				
				iuv_tgt = imageio.imread(os.path.join(data_root, vid_name, method+"_dp_iuv", "frame_{:06d}.png".format(i)))
				iuv_ref = imageio.imread(os.path.join(data_root, vid_name, method+"_dp_iuv", "frame_{:06d}.png".format(i-1)))
				iuv_tgt[:,:,2:3] = iuv_tgt[:,:,2:3]/24.*255.
				iuv_ref[:,:,2:3] = iuv_ref[:,:,2:3]/24.*255.
				iuv_tgt = torch.tensor(iuv_tgt).cuda().float().permute([2,0,1])[None,...]
				iuv_ref = torch.tensor(iuv_ref).cuda().float().permute([2,0,1])[None,...]

				iuv_tgt = F.interpolate(iuv_tgt, [h_tgt,w_tgt], mode="nearest")
				iuv_ref = F.interpolate(iuv_ref, [h_tgt,w_tgt], mode="nearest")
				iuv_flow_fw = flow_model.pred_flow(iuv_tgt, iuv_ref)

				mask = (iuv_tgt[:,2:3]>0).expand_as(iuv_flow_fw)
				
				mse = F.mse_loss(rgb_flow_fw*mask, iuv_flow_fw*mask, reduction="none").sum()/mask.sum()
				mse_list.append(mse.detach().cpu())
		pdb.set_trace()

	mean = torch.mean(torch.stack(mse_list))
	std = torch.std(torch.stack(mse_list))
	print(method)
	print("mean:{}, std:{}".format(mean, std))
		# for i in range(1,len(os.listdir(os.path.join(data_root, vid_name, "images")))+1):
			# img1 = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i)))




