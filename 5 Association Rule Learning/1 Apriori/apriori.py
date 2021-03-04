import pandas as pd

# importing data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None )
transections = [[str(dataset.values[i,j]) for j in range(0,20)] for i in range(0,7501)]

# getting the association rules
from apyori import apriori
rules = apriori(transections, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=5 )

# visualizing th rules
results = list(rules)

# code upgrade by instructor for visualization
listed_rules = []
for i in range(0, len(results)):
    if 'nan' in (str(list(results[i][0])[0]) + str(list(results[i][0])[1])):
        pass
    else:
        listed_rules.append('Rule: ' + str(list(results[i][0])[0]) +
                        ' -> ' + str(list(results[i][0])[1]) +
                        ' \nS: ' + str(round(results[i].support, 4)) +
                        ' \nC: ' + str(round(results[i][2][0].confidence, 4)) +
                        ' \nL: ' + str(round(results[i][2][0].lift, 4)))

# my code for better visualization
my_listed_rules = []
for i in range(0, len(results)):
    s = ''
    b = True
    for j in list(results[i][0]):
        s = s + str(j) + ' - '
    s = s[0:-2]
    s = s + "\nS : " + str( round(results[i].support,4)) 
    s = s + "\nC : " + str( round(results[i][2][0].confidence,4))
    s = s + '\nL : ' + str( round(results[i][2][0].lift,4))
    my_listed_rules.append(s)

# getting the list of associated items
my_list = []
for i in range(0, len(results)):
    my_list.append(list(results[i][0]))










