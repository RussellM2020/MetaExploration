from fileHandling import read
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
prefix = '/home/russellm/rllab/data/local/'

a1 = read(prefix+'4wayTest-plain-seed4/Returns_seed4.csv')
b1 = read(prefix+'4wayTest-plain-seed9/Returns_seed9.csv')
c1 = read(prefix+'4wayTest-plain-seed15/Returns_seed15.csv')


composite = np.vstack((a1[0],b1[0],c1[0]))
addedNoiseMean = np.mean(composite, axis =0)
addedNoiseStd = np.std(composite, axis = 0)
Iterations = range(7)


plt.plot(Iterations, addedNoiseMean, '-r')
plt.fill_between(Iterations, addedNoiseMean-addedNoiseStd, addedNoiseMean+addedNoiseStd,facecolor='r',alpha=0.3)
#print(composite)


for row in composite:
    plt.scatter(Iterations, row)
plt.savefig("4wayplain.png")


# L = [a1,a2,a3,b1,b2,b3]
# for i in range(len(L)):
#     L[i] = np.mean(L[i], axis = 1)[0]
# #print(L)
# plainMean = np.mean([L[0],L[1],L[2]], axis = 0)
# addedNoiseMean = np.mean([L[3],L[4],L[5]], axis = 0) 

# plainStd = np.std([L[0],L[1],L[2]], axis = 0)
# addedNoiseStd = np.std([L[3],L[4],L[5]], axis = 0)

# print(np.shape(plainStd))
# print(plainStd) 


# K = 7
# Iterations = range(7)


# plt.xlabel("Iterations")
# plt.ylabel("Sum of Rewards")
    

# plt.plot(Iterations, plainMean, '-r', label  = 'Normal Obs state')
# plt.plot(Iterations, addedNoiseMean, '-b', label = 'Noise added to obs state')

# plt.fill_between(Iterations, plainMean-plainStd, plainMean+plainStd,facecolor='r',alpha=0.5)
# plt.fill_between(Iterations, addedNoiseMean-addedNoiseStd, addedNoiseMean+addedNoiseStd,facecolor='b',alpha=0.5)


# #plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
# #plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

# plt.axis([0, K-1,0, 20000])
# plt.title("Effect of Adding Correlated Noise in Meta Learning")
# plt.legend()

# #plt.show()
# plt.savefig("SingleRegion_uni.png")

