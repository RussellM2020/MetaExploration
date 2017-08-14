

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import testRollout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import pickle




#prefix = "/home/russellm/rllab/data/local/singleRegion-frameskip8-noNoise-seed4/"

prefix = "/home/russellm/rllab/data/local/4WayComplex-Test-ones-seed"

#exp = "metaLearning_singleRegionGoal_2walls_noNoise/"


def plot(_file, target):

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config = config) as sess:


       
        data = joblib.load(_file)
        policy = data['policy']
        env = data['env']

        rewards = testRollout(env, policy, max_path_length=300,
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
num_itrs = 100
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
percentHistory = []

for itr in range(40,100):
    percent, Image = task(prefix, itr)
    plt.imshow(Image)
    plt.title("Success Rate: "+str(percent)+" percent" )
    percentHistory.append(percent)

    fobj = open("complex4state_metaLearned_ones/Records_40onwards.csv", "wb")
    
    pickle.dump(percentHistory, fobj)
    fobj.close()
    
    plt.savefig("complex4state_metaLearned_ones/itr_"+str(itr)+".png")
    plt.clf()
    







