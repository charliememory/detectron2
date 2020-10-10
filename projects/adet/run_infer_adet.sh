source ~/.bashrc_liqianma

data_path=/esat/dragon/liqianma/datasets/Pose/youtube
out_dir=${data_path}/CondIns_results
mkdir ${out_dir}

# python demo/demo.py \
#     --config-file configs/CondInst/MS_R_101_3x_sem.yaml \
# 	--video-input ${data_path}/youtube_single_cut.mp4 \
# 	--output ${out_dir} \
#     --opts MODEL.WEIGHTS CondInst_MS_R_101_3x_sem.pth

python demo/demo.py \
    --config-file configs/CondInst/MS_R_101_3x_sem.yaml \
	--video-input ${data_path}/youtube_multi_cut.mp4 \
	--output ${out_dir} \
	--confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS ../../pretrain/CondInst_MS_R_101_3x_sem.pth
