import matplotlib, os
matplotlib.use('pdf')
import matplotlib.pyplot as plt

## old
infer_time_base = [0.178,0.224,0.238,0.276,0.306,0.328,0.392,0.432,0.454,0.469,0.590,0.641,0.780,0.739]
infer_time_pure_base = [0.055,0.060,0.063,0.069,0.071,0.075,0.083,0.084,0.087,0.089,0.101,0.109,0.104,0.129]

infer_time_ours = [0.199,0.232,0.229,0.249,0.301,0.332,0.351,0.406,0.420,0.416,0.467,0.459,0.469,0.586]
infer_time_pure_ours = [0.090,0.095,0.100,0.102,0.103,0.102,0.102,0.113,0.115,0.115,0.121,0.120,0.117,0.118]

## new
infer_time_base = [0.145, 0.184, 0.201, 0.244, 0.256, 0.280, 0.333, 0.375, 0.391, 0.395, 0.465, 0.577, 0.564, 0.677]
infer_time_pure_base = [0.055, 0.059, 0.063, 0.069, 0.071, 0.075, 0.083, 0.083, 0.086, 0.087, 0.101, 0.107, 0.104, 0.128]

infer_time_oursNoSparse = [0.326, 0.328, 0.342, 0.354, 0.372, 0.409, 0.429, 0.464, 0.461, 0.465, 0.544, 0.530, 0.553, 0.646]
infer_time_pure_oursNoSparse = [0.205, 0.207, 0.208, 0.209, 0.206, 0.208, 0.207, 0.207, 0.210, 0.206, 0.210, 0.203, 0.210, 0.215]

infer_time_ours = [0.191, 0.201, 0.215, 0.226, 0.267, 0.257, 0.291, 0.349, 0.354, 0.352, 0.420, 0.454, 0.435, 0.51]
infer_time_pure_ours = [0.089, 0.095, 0.098, 0.102, 0.103, 0.102, 0.105, 0.113, 0.114, 0.116, 0.124, 0.122, 0.120, 0.132]

## Draw & Save figure 
# for metric_name in metric_names:

fig = plt.figure(figsize=(6,3)) # ,fontsize=20
x = list(range(1,len(infer_time_base)+1))
plt.plot(x, infer_time_base, label="DP R-CNN DeepLab (Res50)")
# plt.plot(x, infer_time_oursNoSparse, label="Ours w/o Sparse")
plt.plot(x, infer_time_ours, label="Ours (Res50)")
plt.legend(loc='upper left', fontsize='large')
plt.savefig("./tmp/infer_time_two.pdf", format="pdf")  

# fig = plt.figure(figsize=(6,3)) # ,fontsize=20
# x = list(range(1,len(infer_time_base)+1))
# plt.plot(x, infer_time_pure_base, label="DensePose R-CNN")
# # plt.plot(x, infer_time_pure_oursNoSparse, label="Ours w/o Sparse")
# plt.plot(x, infer_time_pure_ours, label="Ours")
# plt.legend(loc='upper left', fontsize='large')
# plt.savefig("./tmp/infer_time_pure_two.pdf", format="pdf")  

 