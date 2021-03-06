

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import testRollout
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import pickle




#prefix = "/home/russellm/rllab/data/local/singleRegion-frameskip8-noNoise-seed4/"

prefix1 = "/home/russellm/rllab/data/local/4way/ones_CorNoise_400/Test-2steps-CorNoise-INIT-seed"
prefix2 = "/home/russellm/rllab/data/local/4wayTest-2steps-CorNoise-seed"

#exp = "metaLearning_singleRegionGoal_2walls_noNoise/"


def plot(_file, target):

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config = config) as sess:


       
        data = joblib.load(_file)
        policy = data['policy']
        env = data['env']

        rewards = testRollout(env, policy, max_path_length=50,
                       animated=False, reset_arg = target)

        #ax = sns.heatmap(path , cmap="YlGnBu")
        #plt.savefig("/home/russellm/MetaPlots/"+exp+"itr_"+str(7*point + itr)+".png")
        #print(rewards)
        return rewards

threshold = 100


def check_success(rewards):
    hits = 0
    total = 0
    #print(rewards)
    for i in rewards:
        total+=1
        if i == threshold:
            hits+=1
    if float(hits/total)>0.5:
        return True
    return False






num_points = 4
num_itrs = 7
num_seeds = 3
num_rollouts_per_file = 100
seeds = ["4/", "9/", "15/"]
targets = [(0.9, 0.9), (0.9, -0.9), (-0.9, -0.9), (-0.9,0.9)]


percentage_History = []

#for itr in range(num_itrs):
#itr = 19
num_success = 0
tar_index=0
image = np.zeros((num_points*num_seeds, num_rollouts_per_file))
imRow = 0


def task(prefix, itr):


    num_success = 0
    tar_index=0
    image = np.zeros((num_points*num_seeds, num_rollouts_per_file))
    imRow = 0





    for point in range(0,num_points):

        print("POINT  " + str(point))
        for seed in seeds:

            print("SEED " + seed)
            _file = prefix+seed+"point_"+str(point)+"/itr_"+str(itr)+".pkl"

            imCol = 0

            for i in range(num_rollouts_per_file):
                print("hello")
                rewards = plot(_file, targets[tar_index])[40:]
    
                if check_success(rewards):
                    num_success+=1
                    image[imRow][imCol] = 1
                imCol+=1
            imRow+=1
        tar_index+=1

    percentage_success = int((100*num_success)/(num_points*num_seeds*num_rollouts_per_file))
    return percentage_success, image
#percentage_History.append(percetage_success)
randHistory = []
metaHistory = []
for itr in range(50):
    randInit, randImage = task(prefix1, itr)
    plt.imshow(randImage)
    plt.title("Success Rate: "+str(randInit)+" percent" )
    
    plt.savefig("meta_learned_initialization_itr "+str(itr)+".png")
    plt.clf()
    meta, metaImage = task(prefix2, itr)
    plt.imshow(metaImage)
    plt.title("Success Rate: "+str(meta)+" percent" )

    plt.savefig("rand_initialization_itr "+str(itr)+".png")
    plt.clf()
    randHistory.append(randInit)
    metaHistory.append(meta)
#print(percentage_History)
#plt.imshow(image)
#plt.title("Success Rate: "+str(percentage_success)+ " percent. (itr_19)")
Iterations = range(0,50)
plt.plot(Iterations, randHistory, "-b", label = "rand Initialization")
plt.plot(Iterations, metaHistory, "-r", label="Meta Initialization")
plt.legend()

plt.savefig("4state_compare.png")


fobj = open("4state_randHistory.csv", "wb")
    
pickle.dump(randHistory, fobj)
fobj.close()

fobj1 = open("4state_metaHistory.csv", "wb")
pickle.dump(metaHistory, fobj1)
fobj1.close()



