from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True
_C.MODEL.RESNETS.BLUR_POOL = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True
_C.MODEL.FCOS.DISABLE_CLASS_LOSS = False 

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.MAX_PROPOSALS = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8 ## original: 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS = 128 
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "GN" #"BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False
_C.MODEL.CONDINST.MASK_BRANCH.NUM_LAMBDA_LAYER = 0
_C.MODEL.CONDINST.MASK_BRANCH.LAMBDA_LAYER_R = 23
_C.MODEL.CONDINST.MASK_BRANCH.USE_ASPP = False
_C.MODEL.CONDINST.MASK_BRANCH.USE_SAN = False
_C.MODEL.CONDINST.SAN_TYPE = "SAN_BottleneckGN" # SAN_BottleneckGN_noInputNormReLU, SAN_BottleneckGN_Gated
_C.MODEL.CONDINST.MASK_BRANCH.USE_ATTN = False
_C.MODEL.CONDINST.ATTN_TYPE = "Spatial_Attn" # SpatialMaxAvg_Attn, SpatialMaxAvg_ChannelMaxAvg_Attn
_C.MODEL.CONDINST.MASK_BRANCH.RESIDUAL_INPUT = False
_C.MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU = False
_C.MODEL.CONDINST.CHECKPOINT_GRAD_NUM = 0
_C.MODEL.CONDINST.v2 = False
_C.MODEL.CONDINST.MASK_BRANCH.TREE_FILTER = False
_C.MODEL.CONDINST.MASK_BRANCH.TREE_FILTER_EMBED_DIM = 16
_C.MODEL.CONDINST.MASK_BRANCH.TREE_FILTER_GROUP_NUM = 16

# ---------------------------------------------------------------------------- #
# CondInst added options
# ---------------------------------------------------------------------------- #
_C.MODEL.NUM_CLASSES = 1 # only background or foreground (person), from densepose MODEL.ROI_HEADS.NUM_CLASSES

_C.MODEL.CONDINST.IUVHead = CN()
_C.MODEL.CONDINST.IUVHead.OUT_CHANNELS = 25*3
_C.MODEL.CONDINST.IUVHead.CHANNELS = 128
_C.MODEL.CONDINST.IUVHead.NORM = "GN" #"BN"
_C.MODEL.CONDINST.IUVHead.NUM_CONVS = 3
_C.MODEL.CONDINST.IUVHead.GT_INSTANCES = False # Use gt instances labels to create relative coords
_C.MODEL.CONDINST.IUVHead.REL_COORDS = False # Use relative coords (2-channels) as input
_C.MODEL.CONDINST.IUVHead.ABS_COORDS = False # Use absolute coords (2-channels) as input
_C.MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER = 0
_C.MODEL.CONDINST.IUVHead.NORM_FEATURES = False
_C.MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS = 0
_C.MODEL.CONDINST.IUVHead.USE_MASK_FEATURES = False
_C.MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES = "none" # none | soft | hard
_C.MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R = 23 # the receptive field for relative positional encoding (23 x 23)
_C.MODEL.CONDINST.IUVHead.NUM_DCN_LAYER = 0
_C.MODEL.CONDINST.IUVHead.DOWN_UP_SAMPLING = False
_C.MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW = False  
_C.MODEL.CONDINST.IUVHead.PARTIAL_CONV = False
_C.MODEL.CONDINST.IUVHead.PARTIAL_NORM = False
_C.MODEL.CONDINST.IUVHead.DISABLE = False
_C.MODEL.CONDINST.IUVHead.NAME = "IUVHead"
_C.MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE = 0
_C.MODEL.CONDINST.IUVHead.USE_AGG_FEATURES = False
_C.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN = False
_C.MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP = False
_C.MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION = False
_C.MODEL.CONDINST.IUVHead.Efficient_Channel_Attention = False
_C.MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION = "none" # none | AfterConv | AfterNorm | AfterRelu
_C.MODEL.CONDINST.IUVHead.RESIDUAL_INPUT = False
_C.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_AFTER_RELU = False
_C.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_LATER = False
_C.MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_CONV = False
_C.MODEL.CONDINST.IUVHead.SKELETON_FEATURES = False
_C.MODEL.CONDINST.IUVHead.GT_SKELETON = False
_C.MODEL.CONDINST.IUVHead.DILATION_CONV = "none"
_C.MODEL.CONDINST.IUVHead.DILATION_CONV_R_MAX = 16
_C.MODEL.CONDINST.IUVHead.PREDICTOR_TYPE = "conv"
_C.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_CONV = False
_C.MODEL.CONDINST.IUVHead.SUBM_CONV = True
_C.MODEL.CONDINST.IUVHead.DROPOUT = False
_C.MODEL.CONDINST.IUVHead.USE_SAN = False

_C.MODEL.CONDINST.FINETUNE_IUVHead_ONLY = False
_C.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV = False
_C.MODEL.INFERENCE_SMOOTH_FRAME_NUM = 0

_C.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_S = False
_C.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON = False
_C.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_S_WEIGHTS = 1.0
_C.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON_WEIGHTS = 1.0
_C.MODEL.CONDINST.AUX_SUPERVISION_BODY_SEMANTICS = False
_C.MODEL.CONDINST.AUX_SUPERVISION_BODY_SEMANTICS_WEIGHTS = 1.0

_C.MODEL.CONDINST.PREDICT_INSTANCE_BODY = False
_C.MODEL.CONDINST.INFER_INSTANCE_BODY = False

_C.MODEL.FCOS.DISABLE_CLASS_LOSS = True

_C.MODEL.TEACHER_STUDENT = False
_C.MODEL.TEACHER_CFG_FILE = "none"
_C.MODEL.TEACHER_WEIGHTS = "none"
_C.MODEL.TEACH_INS_WO_GT_DP = False
_C.MODEL.TEACH_POINT_REGRESSION_WEIGHTS = 10.
_C.MODEL.TEACH_PART_WEIGHTS = 1.0

_C.MODEL.CONDINST.RAND_FLIP = False
_C.MODEL.CONDINST.RAND_SCALE = False
_C.MODEL.CONDINST.RAND_SCALE_GAP = 0.3
_C.MODEL.CONDINST.INFER_TTA_WITH_RAND_FLOW = False
_C.MODEL.CONDINST.INFER_TTA_INSTANCE_MASK = False
_C.MODEL.CONDINST.INFER_INSTANCE_NUM = -1
_C.MODEL.CONDINST.INFER_INSTANCE_NAME = None

_C.SOLVER.ACCUMULATE_GRAD_ITER = 1
_C.SOLVER.FIND_UNUSED_PARAMETERS = True ##
# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN 
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""




