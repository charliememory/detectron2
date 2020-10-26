source ~/.bashrc_liqianma

cfg_name='densepose_CondInst_R_101_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
	--resume \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
    OUTPUT_DIR ./output/${cfg_name}_noS \
    MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0.
    # MODEL.WEIGHTS ./output/${cfg_name}/model_0129999.pth \
    # DATALOADER.NUM_WORKERS 1 \
