from fileHandling import read
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
prefix1 = '/home/russellm/rllab/data/local/MetaTested_Perfectly_Correlated_Noise/'
prefix2 = '/home/russellm/rllab/data/local/MetaTested_randNoise_Obs/'

a1 = read(prefix1+'doubleRegion-randNoise-seed4/Returns_seed4.csv')
b1 = read(prefix1+'doubleRegion-randNoise-seed9/Returns_seed9.csv')
c1 = read(prefix1+'doubleRegion-randNoise-seed15/Returns_seed15.csv')

comp = [a1[0],b1[0],c1[0]]

composite = np.mean(comp, axis=0)

hits=0
for row in composite:
    for item in row:
        if item>100:
            hits+=1

successRate = int((100*hits)/(50*7))

plt.title("Success Rate : " + str(successRate)+ " percent (Threshold is 100)")


ax = sns.heatmap(composite, cmap="YlGnBu")
plt.savefig("doubleRegion_addedNoise.png")

# composite = np.vstack((a1[0],b1[0],c1[0]))
# addedNoiseMean = np.mean(composite, axis =0)
# addedNoiseStd = np.std(composite, axis = 0)
# Iterations = range(7)

# #plt.ylim(0,2500)
# plt.plot(Iterations, addedNoiseMean, '-r')
# plt.fill_between(Iterations, addedNoiseMean-addedNoiseStd, addedNoiseMean+addedNoiseStd,facecolor='r',alpha=0.3)
# #print(composite)


# for row in composite:
#     plt.scatter(Iterations, row)
# plt.savefig("doubleRegion_randNoise.png")


# L = [a1,a2,a3,b1,b2,b3]
# for i in range(len(L)):
#     L[i] = np.mean(L[i], axis = 1)[0]
# #print(L)
# randNoiseMean = np.mean([L[0],L[1],L[2]], axis = 0)
# addedNoiseMean = np.mean([L[3],L[4],L[5]], axis = 0) 

# randNoiseStd = np.std([L[0],L[1],L[2]], axis = 0)
# addedNoiseStd = np.std([L[3],L[4],L[5]], axis = 0)

# print(np.shape(randNoiseStd))
# print(randNoiseStd) 


# K = 7
# Iterations = range(7)


# plt.xlabel("Iterations")
# plt.ylabel("Sum of Rewards")
    

# plt.plot(Iterations, randNoiseMean, '-r', label  = 'Normal Obs state')
# plt.plot(Iterations, addedNoiseMean, '-b', label = 'Noise added to obs state')

# plt.fill_between(Iterations, randNoiseMean-randNoiseStd, randNoiseMean+randNoiseStd,facecolor='r',alpha=0.5)
# plt.fill_between(Iterations, addedNoiseMean-addedNoiseStd, addedNoiseMean+addedNoiseStd,facecolor='b',alpha=0.5)


# #plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
# #plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

# plt.axis([0, K-1,0, 20000])
# plt.title("Effect of Adding Correlated Noise in Meta Learning")
# plt.legend()

# #plt.show()
# plt.savefig("SingleRegion_uni.png")

