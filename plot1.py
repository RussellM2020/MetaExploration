from fileHandling import read
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
prefix = '/home/russellm/rllab/data/local/'

a1 = read(prefix+'doubleRegion-frameskip8-noNoise/Returns_seed4.csv')

res = np.mean(a1, axis = 1)[0]


K = 7
Iterations = range(7)


plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
    

plt.plot(Iterations, res, '-r', label  = 'meta')


#plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
#plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

plt.axis([0, K,0, 10000])
plt.title("meta")
plt.legend()
    
#plt.show()
plt.savefig("meta.png")

