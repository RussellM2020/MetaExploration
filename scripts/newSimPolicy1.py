import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import mazeRollout

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()




prefix = "/home/russellm/rllab/data/local/experiment/baseline/"


exp = "explicitNoise/"


def plot(itr):
    with tf.Session() as sess:


       
        data = joblib.load(prefix+"itr_"+str(itr)+".pkl")
        policy = data['policy']
        env = data['env']

        path = mazeRollout(env, policy, max_path_length=500,
                       animated=True)

        ax = sns.heatmap(path , cmap="YlGnBu")
        plt.savefig("/home/russellm/MetaPlots/"+exp+"Noise_baseline/itr_"+str(itr)+".png")

        sess.close()
        

plot(0)