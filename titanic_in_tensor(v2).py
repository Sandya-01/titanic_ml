import pandas as pd
import tensorflow as tf
import numpy as np
path="D:\\TITANIC ML\\all\\train.csv"
path1="D:\\TITANIC ML\\all\\test.csv"
df=pd.read_csv(path)
df2=pd.read_csv(path1)



headers=df.columns.values
#print(headers)
#print(df.dtypes)


#DATA CLEANSING

#FOR TRAIN DATA
#
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
df.to_csv("train.csv")

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




maximum=df[["Fare"]].max()
print(maximum,11111111111111111111111111111111111111111111)
minimum=df[["Fare"]].min()
print(minimum,22222222222222222222222222222222222222222222222)
df[["Fare"]]=(df[["Fare"]]-minimum)/maximum-minimum
df.to_csv("train.csv")
#for b in df[["Fare"]]:
#    print(b,maximum,minimum)
#    print((b -minimum)/(maximum - minimum))

df2["Age"]=df2["Age"]/df["Age"].max()
df2.to_csv("test.csv")  

# SPLITTING THE TRAIN DATA AND THE TEST DATA

train = df[['Sex','Pclass',"Age"]]
train_data=train[:751] 
#print(train_data)

target=df[["Survived"]]
print(df['Survived'].value_counts())
train_target=target[:751]
test=df[['Sex',"Pclass","Age"]]
test_data=test[751:]
#print(test_data)
test_target = target[751:]
#print(test_data.shape)

# USING TENSOR FLOW

weight=tf.Variable(1.5)
bias=tf.Variable(1.3)
W2=tf.Variable(1.1)
W3=tf.Variable(1.2)

x= tf.placeholder(tf.float32)
desired_op=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)

y = ((weight * x)+( W2 * x2)+ (W3 * x3)) +bias
loss=tf.reduce_mean(tf.square(y-desired_op))
optimizer = tf.train.GradientDescentOptimizer(0.005)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(5555):
    sess.run([train], feed_dict={x2:df[["Age"]][:751],x3:df[["Pclass"]][:751],x:df[["Fare"]][:751],desired_op:df[["Survived"]][:751]} )
    if step % 55 == 0:
        print(step, sess.run([weight,W2,W3,bias])) #print step and value of a & b
test_prediction = sess.run(y, feed_dict={ x2:test_data[["Age"]],x3:test_data[["Pclass"]],x:df[["Fare"]][751:]})

print(len(test_prediction),test_prediction)
t = []
for a in test_prediction:
    #print(a)
    if a[0] < 0.5:
        t.append(0)
    else:
        t.append(1)
print(len(t),len(list(test_target["Survived"])))    
c=0
for i in range(140):
    if t[i] == list(test_target["Survived"])[i]:
        c+=1
print(c/140 *100)        
sess.close()
