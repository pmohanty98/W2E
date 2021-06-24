import pandas as pd
pd.options.mode.chained_assignment = None
import json
import sys
import numpy as np
from scipy import spatial
from sklearn.metrics import mean_absolute_error
from statistics import mean


def listrefiner(list1, list2):
    refinedlist1=[]
    refinedlist2=[]
    for i in range(len(list1)):
        if((list1[i]!=0) and( list2[i]!=0)):
            refinedlist1=list1[i]
            refinedlist2=list2[i]


    tuple=(refinedlist1,refinedlist2)
    #print(tuple)

    return tuple

dishcsv=sys.argv[1]
trainjson=sys.argv[2]
testjson=sys.argv[3]


df = pd.read_csv(dishcsv)
dishidcol=df.iloc[:,0]
data_file=open(trainjson)
traindict = json.load(data_file)
usercol=(traindict.keys())
averagedict={}
traindf = pd.DataFrame(columns=usercol)

traindf.insert(loc=0, column='dish_id', value=dishidcol)

for col in traindf.columns:
    if(col=='dish_id'):
        continue
    traindf[col].values[:] = 0

for i in traindict:
    length=len(traindict[i])
    sum=0
    for j in traindict[i]:
        sum+= int(j[1])
    avrg=sum/length
    averagedict[i] = avrg
    for j in traindict[i]:
        traindf[i][j[0]]= int(j[1])-avrg


data_file=open(testjson)
testdict = json.load(data_file)

tupleofrefinedlists=()
MAElist=[]
MAElistb=[]
pten=[]
ptwenty=[]
rten=[]
rtwenty=[]
for i in testdict:
    y_true = []
    y_pred = []
    rankinglist = []
    for j in testdict[i]:
        customer=i
        dish=j[0]
        label=j[1]
        numerator=0
        denominator=0
        result=0
        for column in traindf:
            if((column=='dish_id') or (column==customer)):
             continue
            if(traindf.at[dish,column]!=0):
                tupleofrefinedlists=listrefiner(traindf[column].tolist(),traindf[i].tolist())
                if not tupleofrefinedlists[0]:
                    continue
                else:
                    result = 1 - spatial.distance.cosine(tupleofrefinedlists[0], tupleofrefinedlists[1])
                    numerator+=result*traindf.at[dish,column]
                    denominator+=result

        if(round(denominator)!=0):
            ratio=(numerator/denominator)
            finalpred=round(averagedict[customer]+ ratio)

            if(finalpred>=5):
                finalpred= 5
            elif(finalpred<=1):
                finalpred=1
        else:
            finalpred = round(averagedict[customer])
        y_true.append(label)
        y_pred.append(finalpred)
        tuple=(finalpred,label,customer,dish)
        rankinglist.append(tuple)
    #print(y_pred)
    #print(y_true)
    #baselinepred=[3]*len(y_pred)
    #print(baselinepred)
    MAE=mean_absolute_error(y_true, y_pred)
    #MAEb=mean_absolute_error(y_true,baselinepred)
    MAElist.append(MAE)
    #MAElistb.append(MAEb)
    #print("MAE: "+str(MAE)[0:7])
    #y_pred = sorted(y_pred,  reverse=True)
    rankinglist=sorted(rankinglist, key=lambda x: x[0],reverse=True)

    prediction_ten = 0
    for i in range(10):
        if ( (rankinglist[i][1] >= 3)):
            prediction_ten += 1

        precisionatten = prediction_ten / 10

    prediction_twenty = 0
    for i in range(20):
        if ((rankinglist[i][1] >= 3)):
            prediction_twenty += 1

        precisionattwenty = prediction_twenty / 20

    x = 0
    for i in range(len(rankinglist)):
        if (  (rankinglist[i][1] >= 3) ):
            x+=1

    recallatten = prediction_ten / x
    recallattwenty = prediction_twenty / x
    pten.append(precisionatten)
    ptwenty.append(precisionattwenty)
    rten.append(recallatten)
    rtwenty.append(recallattwenty)



print("MAE: "+str( mean(MAElist) )[0:7])
#print("Baseline Error: "+str( mean(MAElistb) )[0:7])
print("Precision@10 : " + str(mean(pten))[0:7])
print("Precision@20 : " + str(mean(ptwenty))[0:7])
rtenfinal=mean(rten)
print("Recall@10 : " + str(rtenfinal)[0:7])
rtwentyfinal=mean(rtwenty)
print("Recall@20 : " + str(rtwentyfinal)[0:7])