from fileHandling import read
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
prefix = '/home/russellm/rllab/data/local/'


b1 = read(prefix+'doubleRegion-frameskip2-randNoise-seed9/Returns_seed9.csv')

# b2 = read(prefix+'doubleRegion-frameskip2-uniNoise-seed4/Returns_seed4.csv')
# b3 = read(prefix+'doubleRegion-frameskip2-uniNoise-seed15/Returns_seed15.csv')

b2 = read(prefix+'doubleRegion-frameskip2-randNoise-seed9/goals_seed9.csv')
for row in b1[0]:
    print(row)
# b2 = read(prefix+'doubleRegion-frameskip2-uniNoise-seed4/Goals_seed4.csv')
# b3 = read(prefix+'doubleRegion-frameskip2-uniNoise-seed15/Goals_seed15.csv')

