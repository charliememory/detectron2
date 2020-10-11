source ~/.bashrc_liqianma

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft.yaml \
#     SOLVER.IMS_PER_BATCH 6 SOLVER.BASE_LR 0.005

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft2.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025


# python train_net.py --config-file configs/HRNet/densepose_rcnn_HRFPN_HRNet_w48_s1x.yaml\
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0075 

# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 OUTPUT_DIR ./output/${cfg_name}
#     # MODEL.WEIGHTS ../../pretrain/

cfg_name='densepose_CondInst_R_101_s1x'
python train_net.py --config-file configs/${cfg_name}.yaml \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 OUTPUT_DIR ./output/${cfg_name} \
    --opts MODEL.WEIGHTS ../../pretrain/CondInst_MS_R_101_3x_sem.pth

# python demo/demo.py \
#     --config-file configs/CondInst/MS_R_101_3x_sem.yaml \
# 	--video-input ${data_path}/youtube_multi_cut.mp4 \
# 	--output ${out_dir} \
# 	--confidence-threshold 0.3 \
#     --opts MODEL.WEIGHTS ../../pretrain/CondInst_MS_R_101_3x_sem.pth


# ./run_eval_dp.sh

# ./run_infer_dp.sh

# cd ~/workspace/Gitlab/DensePoseSmooth
# ./run_train_dpsmooth_1.sh
# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft3.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025