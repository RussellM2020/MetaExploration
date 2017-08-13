import pickle
import numpy as np
from matplotlib import pyplot as plt
# fobj = open("complex4state_genPlain.csv", "rb")
# a = pickle.load(fobj)
# print("plain "+str(len(a)))

# fobj.close()

fobj1 = open("4state_metaCor.csv","rb")
fobj2 = open("4state_CorINIT.csv", "rb")

fobj3 = open("simple4state_ones22.csv","rb")
fobj4 = open("simple4state_onesINIT.csv", "rb")

fobj5 = open("simple4state_plain.csv", "rb")
fobj6 = open("simple4state_plainINIT.csv", "rb")

items = [fobj1,fobj2,fobj3,fobj4,fobj5,fobj6]
for i in range(len(items)):
    items[i] = pickle.load(items[i])



items[2] = np.concatenate([items[2], np.ones((50-len(items[2])))*100])

items[3] = np.concatenate([items[3], np.ones((50-len(items[3])))*100])


items[5] = np.concatenate([items[5], np.ones((50-len(items[5])))*25])


Iterations = range(50)

plt.title("Single vs Double Correlation")

plt.plot(Iterations, items[0], '-k', label="Temporally Cor. Meta-trained Init")
plt.plot(Iterations, items[1], '-g', label="Temporally Cor. Random Init")

plt.plot(Iterations, items[2], '-r', label="2D Cor. Meta-trained Init")
plt.plot(Iterations, items[3], '-b', label="2D Cor. Random Init")

plt.plot(Iterations, items[4], '-m', label="Plain Obs. Meta-trained")
plt.plot(Iterations, items[5], '-c', label="Plain Obs. Random Init")

plt.legend()
plt.plot()
plt.savefig("simple4state_noiseAnalysis.png")
