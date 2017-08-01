from fileHandling import read
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
prefix = '/home/russellm/rllab/data/local/'

a1 = read(prefix+'singleRegion-frameskip8-noNoise-seed4/Returns_seed4.csv')
a2 = read(prefix+'singleRegion-frameskip8-noNoise-seed9/Returns_seed9.csv')
a3 = read(prefix+'singleRegion-frameskip8-noNoise-seed15/Returns_seed15.csv')

b1 = read(prefix+'singleRegion-frameskip8-cor2-seed4/Returns_seed4.csv')

b2 = read(prefix+'singleRegion-frameskip8-cor2-seed9/Returns_seed9.csv')
b3 = read(prefix+'singleRegion-frameskip8-cor2-seed15/Returns_seed15.csv')

L = [a1,a2,a3,b1,b2,b3]
for i in range(len(L)):
    L[i] = np.mean(L[i], axis = 1)[0]
print(L)
plain = np.mean([L[0],L[1],L[2]], axis = 0)
addedNoise = np.mean([L[3],L[4],L[5]], axis = 0) 



K = 7
Iterations = range(7)


plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
    

plt.plot(Iterations, plain, '-r', label  = 'Normal Obs state')
plt.plot(Iterations, addedNoise, '-b', label = 'Noise added to obs state')

#plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
#plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

plt.axis([0, K,0, 10000])
plt.title("Effect of Adding Correlated Noise in Meta Learning")
plt.legend()
    
#plt.show()
plt.savefig("cor2.png")

