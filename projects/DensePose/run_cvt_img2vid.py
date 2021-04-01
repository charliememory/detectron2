import imageio, os, tqdm, pdb
import numpy as np
from skimage.transform import rescale, resize

# data_root = "/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/Pose/youtube"
# vid_infos = {"02_clip01":15, 
# 			"04_clip01":16, 
# 			"06_clip01":0, 
# 			"07_clip01":0, 
# 			"08_clip01":0, 
# 			"09_clip01":0, 
# 			"11_clip01":14, 
# 			"12_clip01":26, 
# 			"15_clip01":0, 
# 			"16_clip01":0}
# methods = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.2_smooth2_smoothCoarseSegm",
# 		   "densepose_rcnn_R_50_FPN_DL_s1x"]

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
# # vid_infos = {"05_clip02":0,}
######################
# methods = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_smooth2_smoothCoarseSegm",
# 		   "densepose_rcnn_R_50_FPN_DL_s1x"]

# for vid_name in tqdm.tqdm(sorted(list(vid_infos.keys()))):
# 	w = imageio.get_writer(os.path.join(data_root, vid_name, "out_vid_{}.mp4".format(vid_name)), fps=24)

# 	if int(vid_name.split("_")[0])%2==0:
# 		idx = 0
# 	else:
# 		idx = 1

# 	start_sec = vid_infos[vid_name]
# 	for i in range(start_sec*24+1, (start_sec+20)*24+1):
# 	# for i in range(1,len(os.listdir(os.path.join(data_root, vid_name, "images")))+1):
# 		# img1 = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i)))
# 		img2 = imageio.imread(os.path.join(data_root, vid_name, methods[idx], "frame_{:06d}.jpg".format(i)))
# 		img3 = imageio.imread(os.path.join(data_root, vid_name, methods[1-idx], "frame_{:06d}.jpg".format(i)))
# 		img = np.concatenate([img2,img3], axis=0)

# 		w.append_data(img)
# 	w.close()

#######################
methods = ["densepose_rcnn_R_50_FPN_DL_s1x",
		   "densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3",
		   "densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_smooth2_smoothCoarseSegm",
		   ]

H, W = None, None
w = imageio.get_writer(os.path.join(data_root, "out_comb.mp4"), fps=24)
for vid_name in tqdm.tqdm(sorted(list(vid_infos.keys()))):
	start_sec = vid_infos[vid_name]
	for i in range(start_sec*24+1, (start_sec+20)*24+1):
	# for i in range(1,len(os.listdir(os.path.join(data_root, vid_name, "images")))+1):
		# img1 = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i)))
		img1 = imageio.imread(os.path.join(data_root, vid_name, methods[0], "frame_{:06d}.jpg".format(i)))
		img2 = imageio.imread(os.path.join(data_root, vid_name, methods[1], "frame_{:06d}.jpg".format(i)))
		img3 = imageio.imread(os.path.join(data_root, vid_name, methods[2], "frame_{:06d}.jpg".format(i)))
		if W is None:
			H, W = img1.shape[:2]
		img = np.concatenate([img1,img2,img3], axis=0)
		if img.shape[1]!=W:
			img = (resize(img, (H*3, W), anti_aliasing=True)*255).astype(np.uint8)


		w.append_data(img)
w.close()

# ########################
# data_root = "/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/Pose/youtube"
# vid_infos = {
# 			"03_clip01":0, 
# 			"05_clip01":21, 
# 			"13_clip01":28}
# methods = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_smooth2_smoothCoarseSegm",
# 		   "densepose_rcnn_R_50_FPN_DL_s1x"]
# for vid_name in tqdm.tqdm(sorted(list(vid_infos.keys()))):
# 	w = imageio.get_writer(os.path.join(data_root, vid_name, "out_vid_{}.mp4".format(vid_name)), fps=24)

# 	if int(vid_name.split("_")[0])%2==0:
# 		idx = 0
# 	else:
# 		idx = 1

# 	start_sec = vid_infos[vid_name]
# 	for i in range(start_sec*24+1, (start_sec+10)*24+1):
# 		img1 = imageio.imread(os.path.join(data_root, vid_name, "images", "frame_{:06d}.jpg".format(i)))
# 		img2 = imageio.imread(os.path.join(data_root, vid_name, methods[idx], "frame_{:06d}.jpg".format(i)))
# 		img3 = imageio.imread(os.path.join(data_root, vid_name, methods[1-idx], "frame_{:06d}.jpg".format(i)))
# 		img = np.concatenate([img1,img2,img3], axis=1)

# 		w.append_data(img)
# 	w.close()

