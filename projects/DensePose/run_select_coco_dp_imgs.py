import os, json, pdb, tqdm
from shutil import copyfile 

split = "minival"
json_path = "datasets/coco/annotations/densepose_{}2014.json".format(split)
if "val" in split:
	img_dir = "datasets/coco/val2014"
out_dir = img_dir + "_dp"
os.makedirs(out_dir)

with open(json_path, "r") as f:
	# pdb.set_trace()
	data = json.load(f)
	for sample in tqdm.tqdm(data["images"]):
		file_name = sample["file_name"]
		src = os.path.join(img_dir, file_name)
		dst = os.path.join(out_dir, file_name)
		copyfile(src, dst)
