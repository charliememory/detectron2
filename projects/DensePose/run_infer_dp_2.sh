
source /HPS/HumanBodyRetargeting7/work/For_Liqian/.bashrc_liqianma
export CUDA_VISIBLE_DEVICES=2
# img_path=/esat/dragon/liqianma/datasets/Pose/KUL/liqian_outdoor_horizontal_01/images/frame_000001.jpg

# data_root=/esat/dragon/liqianma/datasets/Pose/KUL/youtube_multi
# data_root=/esat/dragon/liqianma/datasets/Pose/KUL/liqian_outdoor_horizontal_01_fps6_mask
# data_root=/esat/dragon/liqianma/datasets/Pose/KUL/liqian_indoor_vertical_01_mask
# img_dir=$data_root/images

# Dump mode
# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
# img_dir=$data_root/images
# python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir --output $data_root/DP_dump.pkl -v

# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
# img_dir=$data_root/images
# vis_dir=$data_root/DP_vis
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg --smooth_k 0

# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
# img_dir=$data_root/images
# vis_dir=$data_root/DP_vis_smooth2
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg --smooth_k 2

# ## Visualization mode
# vis_dir=$data_root/DP
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg


data_root=/HPS/HumanBodyRetargeting7/work/For_Liqian/datasets/COCO2014/
img_dir=$data_root/val2014_dp

# ## Show mode
# cfg_name='densepose_rcnn_R_50_FPN_DL_s1x_InsSeg_BS2x8'
# model_name=${cfg_name}
# vis_dir=$data_root/${model_name}
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml \
# 						output/${model_name}/model_final.pth \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir   --vis_rgb_img \

## Show mode
cfg_name='densepose_CondInst_R_50_s1x'
model_name=${cfg_name}_SparseInsINNoOverlapResInput_resIUVOnly_GTinsDilated3_10meanUVLoss_5sLoss_BS8_s1x_pretrainCOCOkeypoints
vis_dir=$data_root/${model_name}_flowTTA
mkdir $vis_dir
python apply_net.py show configs/${cfg_name}.yaml \
						output/${model_name}/model_final.pth \
						$img_dir dp_contour \
						--output $vis_dir   --vis_rgb_img  \
						--opts \
					    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
					    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
					    SOLVER.CHECKPOINT_PERIOD 5000 \
					    DATALOADER.NUM_WORKERS 0 \
					    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
					    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
					    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
					    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
					    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
					    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
					    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
					    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
					    MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
					    MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
					    MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
					    MODEL.CONDINST.v2 True \
					    MODEL.FCOS.INFERENCE_TH_TEST 0.3 \
    					MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
					    MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
    					MODEL.CONDINST.INFER_TTA_WITH_RAND_FLOW True
					    # MODEL.INFERENCE_SMOOTH_FRAME_NUM 2\
					    # SOLVER.AMP.ENABLED True \
