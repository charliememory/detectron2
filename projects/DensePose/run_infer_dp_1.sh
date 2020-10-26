source ~/.bashrc_liqianma
# img_path=/esat/dragon/liqianma/datasets/Pose/KUL/liqian_outdoor_horizontal_01/images/frame_000001.jpg

# data_root=/esat/dragon/liqianma/datasets/nerf/nerf_llff_data/fern
data_root=/esat/dragon/liqianma/datasets/Pose/youtube/youtube_multi
# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/youtube_single
# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
img_dir=$data_root/images
# data_root=/esat/dragon/liqianma/datasets/Pose/PoseTrack17
# img_dir=$data_root/images/bonn_5sec/000044_mpii

# ## Visualization mode
# vis_dir=$data_root/densepose_rcnn_R_101_FPN_DL_s1x_1GPU
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml \
# 						output/densepose_rcnn_R_101_FPN_DL_s1x_1GPU/model_final.pth \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg  --smooth_k 0

# vis_dir=$data_root/densepose_rcnn_R_101_FPN_DL_s1x_InsSeg_1GPU
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_s1x_InsSeg.yaml \
# 						output/densepose_rcnn_R_101_FPN_DL_s1x_InsSeg_1GPU/model_final.pth \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg  --smooth_k 0
						

cfg_name=densepose_CondInst_R_50_s1x
# cfg_name=densepose_CondInst_R_101_DL_s1x
# cfg_name=densepose_CondInst_R_101_s1x
# model_name=${cfg_name}_1chSeg_correctBboxHW
# model_name=${cfg_name}_1chSeg
# model_name=${cfg_name}_1chSeg_lambda1
# model_name=${cfg_name}_1chSeg_RelCoordDetach
# model_name=${cfg_name}_1chSeg_RelCoordDetach_pretrainIns
# # model_name=${cfg_name}_1chSeg_10InsLoss_deepIUVHead_RelCoordDetach_pretrainIns
# model_name=${cfg_name}_1chSeg_10InsLoss_deepIUVHead_pretrainIns
# model_name=${cfg_name}_1chSeg_3InsLoss_deepIUVHead
# model_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_RelCoord
model_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1
# model_name=${cfg_name}_1chSeg_deepIUVHead_RelCoordDetach
# # model_name=${cfg_name}_1chSeg_noIUVLoss
vis_dir=$data_root/${model_name}
mkdir $vis_dir
python apply_net.py show configs/${cfg_name}.yaml \
						output/${model_name}/model_0104999.pth \
						$img_dir dp_u,bbox \
						--output $vis_dir/frame_.jpg  --smooth_k 0 \
						--opts \
						MODEL.FCOS.INFERENCE_TH_TEST 0.3 \
    					MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    					MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    					# MODEL.ROI_DENSEPOSE_HEAD.REL_COORDS True \
 
# data_root=/esat/dragon/liqianma/datasets/Pose/youtube/youtube_single
# img_dir=$data_root/images
# vis_dir=$data_root/DP_vis_smooth2
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg  --smooth_k 2

## Dump mode
# python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir --output $data_root/DP_dump.pkl -v


# python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft2.yaml \
# 						/esat/diamond/liqianma/workspace/detectron2/projects/DensePose/output/ft2/model_final.pth \
# 						$img_dir --output $data_root/DP_dump_ft2.pkl -v

##################
# data_root=/esat/dragon/liqianma/datasets/Selfie/FP_and_UP_clean/datasets/Selfie_AdobeStock
# data_root=/esat/dragon/liqianma/datasets/Selfie/FP_and_UP_clean/datasets/Nonselfie_ATR

# for name in Selfie_AdobeStock Nonselfie_Deepfashion Nonselfie_Deepfashion2 \
# 			Selfie_AdobeStock_Flip Nonselfie_Deepfashion_Flip Nonselfie_Deepfashion2_Flip Nonselfie_ATR_Flip
# do
# 	data_root=/esat/dragon/liqianma/datasets/Selfie/FP_and_UP_clean/datasets/${name}
# 	img_dir=$data_root/img

# 	# ## Visualization mode
# 	# vis_dir=$data_root/DPv2_vis
# 	# mkdir $vis_dir
# 	# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 	# 						../../pretrain/densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 	# 						$img_dir dp_u,bbox \
# 	# 						--output $vis_dir/frame_.jpg  

# 	## Dump mode
# 	python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 							../../pretrain/densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 							$img_dir --output $data_root/DPv2_dump.pkl -v
# done

##################

# data_root=/esat/dragon/liqianma/datasets/Pose/KUL/youtube_multi
# img_dir=$data_root/images
# vis_dir=$data_root/DP_vis
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg
						
# data_root=/esat/dragon/liqianma/datasets/Pose/KUL/youtube_multi
# img_dir=$data_root/images
# ## Dump mode
# python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir --output $data_root/DP_dump.pkl -v