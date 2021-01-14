import matplotlib, os
matplotlib.use('pdf')
import matplotlib.pyplot as plt

infer_time_base = [0.178,0.224,0.238,0.276,0.306,0.328,0.392,0.432,0.454,0.469,0.590,0.641,0.780,0.739]
infer_time_pure_base = [0.055,0.060,0.063,0.069,0.071,0.075,0.083,0.084,0.087,0.089,0.101,0.109,0.104,0.129]

infer_time_ours = [0.199,0.232,0.229,0.249,0.301,0.332,0.351,0.406,0.420,0.416,0.467,0.459,0.469,0.586]
infer_time_pure_ours = [0.090,0.095,0.100,0.102,0.103,0.102,0.102,0.113,0.115,0.115,0.121,0.120,0.117,0.118]

## Draw & Save figure 
# for metric_name in metric_names:
fig = plt.figure(figsize=(6,3)) # ,fontsize=20
x = list(range(1,len(infer_time_base)+1))

# plt.plot(x, infer_time_base, label="DensePose R-CNN")
# plt.plot(x, infer_time_ours, label="Ours")
plt.plot(x, infer_time_pure_base, label="DensePose R-CNN")
plt.plot(x, infer_time_pure_ours, label="Ours")

plt.legend(loc='upper left', fontsize='large')
plt.savefig("./tmp/infer_time_pure.pdf", format="pdf")  

