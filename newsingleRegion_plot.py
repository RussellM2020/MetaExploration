from fileHandling import read
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
prefix = '/home/russellm/rllab/data/local/'

a1 = read(prefix+'singleRegion-frameskip2-addedNoise/Returns_seed9.csv')
a1 = np.mean(a1, axis = 1)[0]
#a2 = read(prefix+'singleRegion-frameskip8-noNoise-seed9/Returns_seed9.csv')
#a3 = read(prefix+'singleRegion-frameskip8-noNoise-seed15/Returns_seed15.csv')

b1 = read(prefix+'singleRegion-frameskip2-noNoise/Returns_seed9.csv')
b1 = np.mean(b1, axis =1)[0]
#b2 = read(prefix+'singleRegion-frameskip8-allNoise-seed9/Returns_seed9.csv')
#b3 = read(prefix+'singleRegion-frameskip8-allNoise-seed15/Returns_seed15.csv')

# L = [a1,a2,a3,b1,b2,b3]
# for i in range(len(L)):
#     L[i] = np.mean(L[i], axis = 1)[0]
# print(L)
# plain = np.mean([L[0],L[1],L[2]], axis = 0)
# addedNoise = np.mean([L[3],L[4],L[5]], axis = 0) 



K = 7
Iterations = range(7)


plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
    
plt.plot(Iterations, a1, '-b', label = 'Noise added to obs state')
plt.plot(Iterations, b1, '-r', label  = 'Normal Obs state')


#plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
#plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

plt.axis([0, K,0, 20000])
plt.title("Effect of Adding Correlated Noise in Meta Learning")
plt.legend()
    
#plt.show()
plt.savefig("newMeta1.png")

