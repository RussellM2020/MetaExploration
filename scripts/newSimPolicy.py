import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import mazeRollout

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()




prefix = "/home/russellm/rllab/data/local/singleRegion-frameskip8-noNoise-seed4/"


exp = "metaLearning_singleRegionGoal_2walls_noNoise/"


def plot(point, itr):
    with tf.Session() as sess:


       
        data = joblib.load(prefix+"point_"+str(point)+"/itr_"+str(itr)+".pkl")
        policy = data['policy']
        env = data['env']

        path = mazeRollout(env, policy, max_path_length=500,
                       animated=True)

        ax = sns.heatmap(path , cmap="YlGnBu")
        plt.savefig("/home/russellm/MetaPlots/"+exp+"itr_"+str(7*point + itr)+".png")

        sess.close()
        

plot(10,3)