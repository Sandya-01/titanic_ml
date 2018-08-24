"""
AUTHOR:
 #6729 5883 5291
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
#import random
path="D:\\TITANIC ML\\all\\train.csv"
path1="D:\\TITANIC ML\\all\\test.csv"
df=pd.read_csv(path)
df2=pd.read_csv(path1)

##print(df.loc[0])#to get the column names which is located in the first row
#
headers=df.columns.values
print(headers)
print(df.dtypes)
#print(df.describe(include="all"))
#statistical_summary=df[["Survived","Age"]].describe()
#print(statistical_summary)
#sort=df.sort_values(["Age"],ascending=False)
#print(sort["Age"].head(1))
#print(df["Age"].max())

#DATA CLEANSING

#FOR TRAIN DATA

df1 = df.fillna(-1)
count = 0
sum1 = 0
for age in df1["Age"]:
    if age == -1:
        pass
#        print(age)

    else:
        sum1 += age
        count +=1
        mean=sum1//count
replacement=df["Age"].fillna(mean,inplace =True)
#print(df["Age"])
#df.to_csv("train.csv")

for gender in df["Sex"]:
    if gender=="male":
        df["Sex"]=df["Sex"].replace("male",0)
    else:
        df["Sex"]=df["Sex"].replace("female",1)

df.to_csv("train.csv")

# FOR TEST DATA
sum2,count2=0,0
for age1 in df2["Age"]:
    if age1==np.nan:
        continue
    else:
        sum2 += age
        count2 +=1
        mean2=sum2//count2
#print(mean2)
replacing=df2["Age"].replace(np.nan,mean2,inplace=True)
#print(df2["Age"])

for gend in df2["Sex"]:
    if gend=="male":
        df2["Sex"]=df2["Sex"].replace("male",0)
    else:
        df2["Sex"]=df2["Sex"].replace("female",1)
    

df2.to_csv("test.csv")

# NBORMALIZING THE DATA

df["Age"]=df["Age"]/df["Age"].max()
#df["Pclass"]=df["Pclass"]/df["Pclass"].max()   

df.to_csv("train.csv")

df2["Age"]=df2["Age"]/df["Age"].max()
df2.to_csv("test.csv")  

# SPLITTING THE TRAIN DATA AND THE TEST DATA

train = df[['Sex','Age']]
train_data=train[:751] 
print(train_data.shape)

target=df[["Survived"]]
train_target=target[:751]

#test=df[['Sex','Age']]
test_data=train[751:]
test_target = target[751:]
print(test_data.shape)

#LOGISTIC REGRESSION

logreg = LogisticRegression()
print(logreg.fit(train_data,train_target))
Y_pred = logreg.predict(test_data)
acc_log = round(logreg.score(test_data,test_target) * 100, 2)
print acc_log
#print test_target
#print Y_pred    
#print(df.Age)
#print(df.Age.round(decimals=2))
#print(round(70.23456))


#for valu1 in df2["Age"]:
#    normal=df2["Age"]/df2["Age"].max()
#print(normal)

        


