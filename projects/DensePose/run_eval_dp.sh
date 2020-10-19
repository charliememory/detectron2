source ~/.bashrc_liqianma

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft.yaml \
#     --eval-only MODEL.WEIGHTS output/model_0129999.pth


cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${cfg_name}/model_final.pth \

# # cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x_InsSeg'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${cfg_name}/model_final.pth \