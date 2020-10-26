source ~/.bashrc_liqianma
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

data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
img_dir=$data_root/images
vis_dir=$data_root/DP_vis
mkdir $vis_dir
python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
						../../pretrain/densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
						$img_dir dp_contour,bbox \
						--output $vis_dir/frame_.jpg --smooth_k 0

data_root=/esat/dragon/liqianma/datasets/Pose/youtube/liqian01
img_dir=$data_root/images
vis_dir=$data_root/DP_vis_smooth2
mkdir $vis_dir
python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
						../../pretrain/densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
						$img_dir dp_contour,bbox \
						--output $vis_dir/frame_.jpg --smooth_k 2

# ## Visualization mode
# vis_dir=$data_root/DP
# mkdir $vis_dir
# python apply_net.py show configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml \
# 						densepose_rcnn_R_101_FPN_DL_WC1_s1x.pkl \
# 						$img_dir dp_contour,bbox \
# 						--output $vis_dir/frame_.jpg

