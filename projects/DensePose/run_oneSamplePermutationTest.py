import imageio, os, tqdm, pdb
import numpy as np

def oneSamplePermutationTest(x, nsim=10**6):
    n = len(x)
    x = np.reshape(x, (n,1))
    dbar = np.mean(x)
    #print(dbar)
    absx = np.absolute(x)
    mn = np.random.choice([-1.0,1.0], size=(n,nsim))
    z = np.mean(mn*absx, axis=0)
    pval = (np.sum(z>=np.absolute(dbar)) + np.sum(z<=-np.absolute(dbar))) / float(nsim)
    return pval

# diff_list = [-5,-5,-9,-9,-9,-7,-7,-9,-9,-9,-7]
# diff_list = [-17,-19,-19,-21,-19,-19,-19,-21,-21,-19,-15,]
# prefer_oriDP = [3,2,2,2,2,2,2,2,1,2,5]
# user_num = 27
prefer_oriDP = [3,2,2,2,2,2,2,2,1,2,5]
user_num = 30 #29
ratio_oriDP = np.array(prefer_oriDP)/user_num
print(np.mean(ratio_oriDP), np.std(ratio_oriDP))
diff_list = [-(user_num-2*n) for n in prefer_oriDP] #[-21,-23,-23,-23,-23,-23,-23,-23,-25,-23,-17]
pval = oneSamplePermutationTest(diff_list, nsim=10**6)
print(pval)