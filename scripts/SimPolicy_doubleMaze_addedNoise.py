import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import testRollout

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()




#prefix = "/home/russellm/rllab/data/local/singleRegion-frameskip8-noNoise-seed4/"

prefix = "/home/russellm/rllab/data/local/MetaTested_Perfectly_Correlated_Noise/doubleRegion-randNoise-seed"
#exp = "metaLearning_singleRegionGoal_2walls_noNoise/"


def plot(_file, target):

    tf.reset_default_graph()

    with tf.Session() as sess:


       
        data = joblib.load(_file)
        policy = data['policy']
        env = data['env']

        rewards = testRollout(env, policy, max_path_length=600,
                       animated=False, reset_arg = target)

        #ax = sns.heatmap(path , cmap="YlGnBu")
        #plt.savefig("/home/russellm/MetaPlots/"+exp+"itr_"+str(7*point + itr)+".png")
        print(rewards)
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






num_points = 50
num_itrs = 7
num_seeds = 3
num_rollouts_per_file = 10
seeds = ["4/", "9/", "15/"]


test_num_goals = 50
np.random.seed(1)
x_vals1 = np.random.uniform(-0.9, -0.5, size = int(test_num_goals/2))
x_vals2 = np.random.uniform(0.5, 0.9, size = int(test_num_goals/2))

x_vals = np.concatenate([x_vals1, x_vals2])
np.random.shuffle(x_vals)
        
y_vals = np.random.uniform(-0.9, -0.2, size = test_num_goals)
        
            
targets = np.column_stack((x_vals,y_vals))



percentage_History = []

#for itr in range(num_itrs):
itr = 6
num_success = 0
tar_index=0
image = np.zeros((12, 10))
imRow = 0

for point in range(0,num_points):
    
    print("POINT  " + str(point))
    for seed in seeds:
        
        print("SEED " + seed)
        _file = prefix+seed+"point_"+str(point)+"/itr_"+str(itr)+".pkl"

        imCol = 0
        
        for i in range(num_rollouts_per_file):

            rewards = plot(_file, targets[tar_index])[580:]
            
            if check_success(rewards): 

                num_success+=1
                image[imRow][imCol] = 1
            imCol+=1
        imRow+=1
    tar_index+=1

percentage_success = int((100*num_success)/(num_points*num_seeds*num_rollouts_per_file))
#percentage_History.append(percetage_success)


#print(percentage_History)
plt.imshow(image)
plt.title("Success Rate: "+str(percentage_success)+ " percent. (itr_6)")
plt.savefig("doubleMaze_allNoise.png")



