import json
import pandas as pd
import math
import numpy as np
import sys
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_error
from statistics import mean
from statistics import median
from statistics import mode

"""
def similarity(list1, list2):
    x=0
    for i in range(len(list1)):
        if((list1[i]==list2[i]) and( list1[i]==1)):
            x=x+1

    size=len(list1)

    return (x)/ size
"""
dishcsv=sys.argv[1]
testjson=sys.argv[2]
df = pd.read_csv(dishcsv)
K=int(sys.argv[3])
cluster_dataframe = df.sample(n=K, random_state=25550)

clusterdict={}
cantsay=[]

df['ID'] = (df.index)
df['belongs to'] = ''
cluster_dataframe['ID'] = cluster_dataframe.index
cluster_dataframe['belongs to'] = ''
copyofcdf = 0
for i in cluster_dataframe['ID']:
    clusterdict[i]=[]
i=0
while (cluster_dataframe.equals(copyofcdf) == False):
    #print("iteration:"+str(i))
    i+=1
    copyofcdf = cluster_dataframe.copy()
    for index1, dfinstance in df.iterrows():
        decidinglist = []
        item=dfinstance.tolist()
        item.pop(0)
        item.pop(0)
        item.pop(len(item)-1)
        item.pop(len(item) - 1)

        for index2, cdfinstance in cluster_dataframe.iterrows():
            seed = cdfinstance.tolist()
            seed.pop(0)
            seed.pop(0)
            seed.pop(len(seed) - 1)
            seed.pop(len(seed) - 1)
            result=sum([x*y for x,y in zip(item,seed)])
            result=result/len(item)
            #print(result)
            tuple = (result, cdfinstance["ID"])
            decidinglist.append(tuple)

        decidinglist.sort(key=lambda tup: tup[0], reverse=True)
        if(decidinglist[0][0]!=0):
            df.at[index1, "belongs to"] = decidinglist[0][1]
        else:
            if index1 not in cantsay:
                cantsay.append(index1)

    for index3, cdfinstance in cluster_dataframe.iterrows():
        partition = df.loc[df["belongs to"] == cdfinstance["ID"]]

        for cols in partition.columns:
            if((cols!="dish_id") and (cols!="dish_name") and (cols!="ID") and (cols!="belongs to")):
                Median_x = partition[cols].median()
                #print(Mean_x)
                if(Median_x>=0.5):
                    Median_x=1
                else:
                    Median_x=0
                cluster_dataframe.at[index3, cols] = Median_x

data_file=open(testjson)
testdict = json.load(data_file)

MAElist=[]
MAElistb=[]
pten=[]
ptwenty=[]
rten=[]
rtwenty=[]
for i in testdict:
    metrictracker=[]
    y_true = []
    y_pred = []
    for j in testdict[i]:
        customer=i
        dish=j[0]
        label=j[1]
        clustermembership=df.at[int(dish),"belongs to"]
        if(clustermembership==''):
            continue
        tuple=(dish,label)
        clusterdict[clustermembership].append(tuple)
    for k in clusterdict:
        average_rating_percluster=0
        candidatelist=[]
        for a in clusterdict[k]:
            #average_rating_percluster+=a[1]
            candidatelist.append(a[1])
        if(len(clusterdict[k])!=0):
            average_rating_percluster=mean(candidatelist)
            #average_rating_percluster=average_rating_percluster/len(clusterdict[k])

            #scorelength=(average_rating_percluster,len(clusterdict[k]))
            #metrictracker.append(scorelength)
        for l in clusterdict[k]:
            scorelength = (average_rating_percluster,l[1])
            metrictracker.append(scorelength)
            y_true.append(l[1])
            y_pred.append(round(average_rating_percluster))


    MAE = mean_absolute_error(y_true, y_pred)
    #baselinepred = [3] * len(y_pred)
    #MAEb = mean_absolute_error(y_true, baselinepred)
    #MAElistb.append(MAEb)
    MAElist.append(MAE)
    metrictracker=sorted(metrictracker, key=lambda x: x[0],reverse=True)
    a=0
    prediction_ten = 0
    while (a<10):
        if ((metrictracker[a][1] >= 3) ):
            prediction_ten += 1
        a += 1

    precisionatten = prediction_ten / 10
    a=0
    prediction_twenty = 0
    while (a<20):
        if ( (metrictracker[a][1] >= 3) ):
            prediction_twenty += 1
        a += 1

    precisionattwenty = prediction_twenty / 20
    x = 0
    for i in range(len(metrictracker)):
        if ( (metrictracker[i][1] >= 3) ):
            x += 1

    if(x==0):
        recallatten=0
        recallattwenty=0
    else:
        recallatten = prediction_ten / x
        recallattwenty = prediction_twenty / x

    pten.append(precisionatten)
    ptwenty.append(precisionattwenty)
    rten.append(recallatten)
    rtwenty.append(recallattwenty)

    for item in clusterdict:
        clusterdict[item].clear()

"""
WC_SSE = 0
for index4, cdfinstance in cluster_dataframe.iterrows():
    partition = df.loc[df["belongs to"] == cdfinstance["ID"]]
    seed = cdfinstance.tolist()
    seed.pop(0)
    seed.pop(0)
    seed.pop(len(seed) - 1)
    seed.pop(len(seed) - 1)


    for index5, pdfinstance in partition.iterrows():

            item = pdfinstance.tolist()
            item.pop(0)
            item.pop(0)
            item.pop(len(item) - 1)
            item.pop(len(item) - 1)
            d=1- ((sum([x*y for x,y in zip(seed,item)]))/len(item))
            WC_SSE=WC_SSE+ (d)**2



BC_SSE = 0
print(K)
print("WC_SSE=" + str(WC_SSE))

for i in range(len(cluster_dataframe)):

        y= cluster_dataframe.iloc[i].tolist()
        y.pop(0)
        y.pop(0)
        y.pop(len(y) - 1)
        y.pop(len(y) - 1)

        for j in range(i + 1, len(cluster_dataframe)):
            x = cluster_dataframe.iloc[j].tolist()
            x.pop(0)
            x.pop(0)
            x.pop(len(x) - 1)
            x.pop(len(x) - 1)

            d=1- ((sum([x*y for x,y in zip(x,y)]))/len(x))
            BC_SSE=BC_SSE+ (d)**2
            
print("BC_SSE=" + str(BC_SSE))
"""

print("MAE: "+str( mean(MAElist) )[0:7])
#print("Baseline Error: "+str( mean(MAElistb) )[0:7])
print("Precision@10 : " + str(mean(pten))[0:7])
print("Precision@20 : " + str(mean(ptwenty))[0:7])
rtenfinal=mean(rten)
print("Recall@10 : " + str(rtenfinal)[0:7])
rtwentyfinal=mean(rtwenty)
print("Recall@20 : " + str(rtwentyfinal)[0:7])


