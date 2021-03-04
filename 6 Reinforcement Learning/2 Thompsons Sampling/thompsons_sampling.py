#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#apply thompsons sampling model
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
no_of_1 = [0] * d
no_of_0 = [0] * d 
for n in range(0,N):
    max_random = ad =  0
    for i in range(0,d):
        random_draw = random.betavariate( no_of_1[i] + 1 , no_of_0[i] + 1 )
        if random_draw > max_random:
            max_random = random_draw
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward = total_reward + reward
    if reward == 1:
        no_of_1[ad] = no_of_1[ad] + 1
    else:
        no_of_0[ad] = no_of_0[ad] + 1

# plotting of histogram of ads selected
plt.hist(ads_selected)
plt.title('histogram for thompsonsmodel')
plt.xlabel('ad')
plt.ylabel('no. of clicks')
plt.show()














