source ~/.bashrc_liqianma

python infer.py --model models/raft-sintel.pth \
				--seq_img_dir /esat/dragon/liqianma/datasets/Pose/KUL/youtube_single/images \
				--backward_flow  #--small --mixed_precision  --alternate_corr

# python infer.py --model models/raft-kitti.pth \
# 				--seq_img_dir /esat/dragon/liqianma/datasets/Pose/KUL/youtube_single/images \
# 				--backward_flow  #--small --mixed_precision  --alternate_corr