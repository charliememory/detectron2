# import contextlib
# from pycocotools.coco import COCO
import os, pdb
import numpy as np
from densepose.data.datasets.coco import *


# def filter_out_no_densepose(img_list):
#     valid_img_list = []
#     for img in img_list:
#         if 'has_no_densepose' in img.keys():
#             # Ignoring frames with no densepose supervision.
#             continue
#         else:
#             valid_img_list.append(img)
#     valid_img_ids = sorted([img['id'] for img in valid_img_list])
#     return valid_img_ids, valid_img_list


# json_file = "/esat/dragon/liqianma/datasets/Pose/COCO2014/densepose_coco_2014_minival.json"
# with contextlib.redirect_stdout(io.StringIO()):
#     coco_api = COCO(json_file)

# img_ids = sorted(coco_api.imgs.keys())
# pdb.set_trace()
# # imgs is a list of dicts, each looks something like:
# # {'license': 4,
# #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
# #  'file_name': 'COCO_val2014_000000001268.jpg',
# #  'height': 427,
# #  'width': 640,
# #  'date_captured': '2013-11-17 05:57:24',
# #  'id': 1268}

# imgs = coco_api.loadImgs(img_ids)
# # pdb.set_trace()

# ##### MLQ added #####
# ## Filter out unlabeled train/val data in densepose-track 
# img_ids, imgs = filter_out_no_densepose(imgs)




# # anns is a list[list[dict]], where each dict is an annotation
# # record for an object. The inner list enumerates the objects in an image
# # and the outer list enumerates over images.
# anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
# _verify_annotations_have_unique_ids(annotations_json_file, anns)
# dataset_records = _combine_images_with_annotations(dataset_name, image_root, imgs, anns)


data_root = "../../datasets/coco"
dataset_name = "densepose_minival2014"
annotations_fpath = os.path.join(data_root, "annotations/{}.json".format(dataset_name))
image_root = os.path.join(data_root, "val2014")
# annotations_fpath = maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
# images_root = maybe_prepend_base_path(datasets_root, dataset_data.images_root)

## Load annos
def load_annotations():
    return load_coco_json(
        annotations_json_file=annotations_fpath,
        image_root=image_root,
        dataset_name=dataset_data.name,
    )
dataset_records = load_coco_json(annotations_fpath, image_root, dataset_name)

## Print statistic info
ins_num = [len(record['annotations']) for record in dataset_records]
ins_num = np.array(ins_num)
img_num_of_diff_ins_num = [(ins_num==i).sum() for i in range(1,ins_num.max()+1)]
print("max_instance_num:{}, mean_instance_num:{}".format(ins_num.max(), ins_num.mean()))
print("img_num of {}-{} ins_num:".format(ins_num.min(),ins_num.max()), img_num_of_diff_ins_num)

## Split annos according to ins_num
for num in range(1,ins_num.max()+1):
	dataset_records_chunk = [record for record in dataset_records if len(record['annotations'])==num]
	
pdb.set_trace()
print()

## person num: 1~4 | 5~8 | 8~12 | 12~16 | 16~20




