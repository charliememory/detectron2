source ~/.bashrc_liqianma

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft.yaml \
#     --eval-only MODEL.WEIGHTS output/model_0129999.pth


# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${cfg_name}/model_final.pth \
# 	MODEL.FCOS.INFERENCE_TH_TEST 0.3 \

# # cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x_InsSeg'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${cfg_name}_1GPU/model_final.pth \


cfg_name='densepose_CondInst_R_50_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0117999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    # > ./output/${mode_name}/eval_log.txt
    # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \


cfg_name='densepose_CondInst_R_50_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_RelCoord
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0114999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
	MODEL.ROI_DENSEPOSE_HEAD.REL_COORDS True \
    # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    # > ./output/${mode_name}/eval_log.txt
    # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
