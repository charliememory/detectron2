import torch, pdb

path = "./hrnetv2_w48_imagenet_pretrained.pth"
save_path = "./hrnetv2_w48_imagenet_pretrained_renamekeys.pth"
pretrained_dict = torch.load(path)
pdb.set_trace()
pretrained_dict = {"backbone.bottom_up."+k: v for k, v in pretrained_dict.items()}
torch.save(pretrained_dict,save_path)