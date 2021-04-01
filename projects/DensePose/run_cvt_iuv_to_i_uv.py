import imageio, os, tqdm, pdb
import numpy as np

data_root = "/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/Pose/MPI_CG/Weipeng_train"

model_names = ["densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_dp_iuv",
"densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints_1smooth_dpMaskInstance_dilate0_th0.3_smooth2_smoothCoarseSegm_dp_iuv",
"densepose_rcnn_R_50_FPN_DL_s1x_dp_iuv"]

for model_name in model_names:
	src_dir = os.path.join(data_root, model_name)
	I_dir = src_dir.replace("dp_iuv", "dp_i")
	if not os.path.exists(I_dir):
		os.makedirs(I_dir)
	for filename in tqdm.tqdm(os.listdir(src_dir)):
		UVI = imageio.imread(os.path.join(src_dir,filename))
		I = UVI[:,:,-1]
		imageio.imwrite(os.path.join(I_dir, filename), I)
