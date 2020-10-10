source ~/.bashrc_liqianma

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft.yaml \
#     --eval-only MODEL.WEIGHTS output/model_0129999.pth


# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft2.yaml \
#     --eval-only MODEL.WEIGHTS ./model_final_f359f3.pkl OUTPUT_DIR output/model_final_f359f3_posetrackVal

python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft2.yaml \
    --eval-only MODEL.WEIGHTS output/model_0024999.pth OUTPUT_DIR output/ft2_posetrackVal

python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft2.yaml \
    --eval-only MODEL.WEIGHTS output/model_final.pth OUTPUT_DIR output/ft2_posetrackVal