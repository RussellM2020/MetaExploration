import scipy.ndimage as sp
import numpy as np
from matplotlib import pyplot as plt


Iterations = range(100)
noise = np.random.normal(0,1,(4,100))
#variance = np.var(noise, axis=0)
#noise = noise / np.sqrt(variance)



def plotNoise(noise, filename):
    for row in noise:
        plt.plot(Iterations, row)
    plt.savefig(filename)
    plt.clf()

def correlate(noise, var):
    for i in range(4):
        noise[i,:] = sp.filters.gaussian_filter(noise[i,:], var)
        

    variance = np.var(noise, axis=0)
    noise = noise / np.sqrt(variance)
    return noise




plotNoise(noise, "initial.png")
noise1 = correlate(noise, 0.1)
plotNoise(noise1, "correlated_1.png")

noise2 = correlate(noise, 1.5)
plotNoise(noise2, "correlated_9.png")

noise4 = correlate(noise, 10)
plotNoise(noise4, "doubly_correlated.png")

noise5 = correlate(noise, 50)
print(noise5[:,0])
print(noise5[:,1])
plotNoise(noise5, "doubly_correlated_1.png")

