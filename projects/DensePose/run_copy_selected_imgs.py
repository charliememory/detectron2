import os, shutil, pdb

data_root = "/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/COCO2014"
img_dir_selected = os.path.join(data_root, "val_imgs_selected")
dir_name = "val2014"
# dir_name = "densepose_rcnn_R_50_FPN_DL_s1x_InsSeg_BS2x8"
# dir_name = "densepose_CondInst_R_50_s1x_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints"
src_dir = os.path.join(data_root, dir_name)
dst_dir = os.path.join(data_root, "selected_"+dir_name)
if not os.path.exists(dst_dir):
	os.makedirs(dst_dir)

for file_name in os.listdir(img_dir_selected):
	shutil.copyfile(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))
