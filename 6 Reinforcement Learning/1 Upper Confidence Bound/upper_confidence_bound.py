#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#prepare the upper confidence bound 
import math
N = 10000
d = 10
ad_selected = []
no_of_selection = [0] * d
sum_of_rewards = [0] * d
for n in range(0,N):
    max_upper_bound = ad = 0
    if n < 10:
        ad = n
    else:
        for i in range(0,d):
            avg_rewards = sum_of_rewards[i] / no_of_selection[i]
            delta = math.sqrt(3/2 * math.log(n+1) / no_of_selection[i])
            upper_bound =  avg_rewards + delta
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
    no_of_selection[ad] = no_of_selection[ad] + 1
    sum_of_rewards[ad] = sum_of_rewards[ad] + dataset.values[n,ad]
    ad_selected.append(ad)
total_rewards = 0
for i in sum_of_rewards:
    total_rewards = total_rewards + i

#plotting histogram of ads selected
plt.hist(ad_selected)
plt.title('frequencies of ads selected')
plt.xlabel('ad')
plt.ylabel('no. of selections')
plt.show()

# bar graph for the ad selected
plt.bar( np.arange(0,10),sum_of_rewards)
plt.show()






