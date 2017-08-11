import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()




#prefix = "/home/russellm/rllab/data/local/singleRegion-frameskip8-noNoise-seed4/"

prefix = "/home/russellm/rllab/data/local/4wayTest-plain-seed"
#exp = "metaLearning_singleRegionGoal_2walls_noNoise/"


def plot(_file):

    tf.reset_default_graph()

    with tf.Session() as sess:


       
        data = joblib.load(_file)
        policy = data['policy']
        env = data['env']

        rewards = rollout(env, policy, max_path_length=50,
                       animated=False)

        #ax = sns.heatmap(path , cmap="YlGnBu")
        #plt.savefig("/home/russellm/MetaPlots/"+exp+"itr_"+str(7*point + itr)+".png")
        
        return rewards

threshold = 100


def check_success(rewards):
    hits = 0
    total = 0
    for i in rewards:
        total+=1
        if i > threshold:
            hits+=1
    if float(hits/total)>0.5:
        return 1
    return 0 






num_points = 4
num_itrs = 7
num_seeds = 3
seeds = ["4/", "9/", "15/"]


percentage_History = []

# for itr in range(num_itrs):

#     num_success = 0

#     for point in range(num_points):
#         for seed in seeds:
num_success = 0
point = 2
seed = "4/"
itr=5
        
_file = prefix+seed+"point_"+str(point)+"/itr_"+str(itr)+".pkl"
rewards = plot(_file)
print(rewards)
num_success+=check_success(rewards)

percetage_success = float(num_success/(num_points*num_seeds))
percentage_History.append(percetage_success)


print(percentage_History)



