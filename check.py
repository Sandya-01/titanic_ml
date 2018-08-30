import pandas as pd
import tensorflow as tf
import numpy as np
path="D:\\TITANIC ML\\all\\train.csv"
path1="D:\\TITANIC ML\\all\\test.csv"
df=pd.read_csv(path)
df2=pd.read_csv(path1)



headers=df.columns.values
print(headers)
print(df.dtypes)


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

maximum=df[["Fare"]].max()
minimum=df[["Fare"]].min()

df[["Fare"]]=(df[["Fare"]]-minimum)/(maximum-minimum)


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

train = df[['Sex',"Fare","Age"]]
train_data=train[:751] 
print(train_data)

target=df[["Survived"]]
print(df['Survived'].value_counts())
train_target=target[:751]
print (train_target)
test=df[['Sex',"Fare","Age"]]
test_data=test[751:]
test_target = target[751:]

#print(test_data.shape)

# USING TENSOR FLOW


training_epochs = 5555 *2
display_step = 30
batch_size = 32


n_hidden1=20
n_hidden2=10
n_inputs=3
n_class=1


weight={ 'h1': tf.Variable(tf.random_normal([n_inputs,n_hidden1])),
                       'h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
                       'out':tf.Variable(tf.random_normal([n_hidden2,n_class]))}
bias={'b1':tf.Variable(tf.random_normal([n_hidden1])),'b2':tf.Variable(tf.random_normal([n_hidden2])),'op':tf.Variable(tf.random_normal([n_class]))}


x= tf.placeholder(tf.float32)
desired_op=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
keep_prob=tf.placeholder("float")

def multilayer_perceptron(x,weight,bias):

    layer_1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
   # print(tf.add(tf.matmul(x, weight['h1']), bias['b1']))
   # print("layer1: ",layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), bias['b2'])
    #print("layer2: ",layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weight['out']) + bias['op']
    print("out layer: ",out_layer)
    return out_layer
logits = multilayer_perceptron(x,weight,bias)
optimizer = tf.train.GradientDescentOptimizer(0.0009)
loss = tf.reduce_mean(tf.square(logits-y))  # y--> predicted output  &  d--> original output

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(training_epochs):
#        avg_cost = 0.0
    c = sess.run([train], 
                            feed_dict={
                                x: train_data, 
                                y: train_target
                                
                            })
#            avg_cost += c / total_batch
           
    if step % 555 == 0:
        print(step) #print step and value of a & b

#            if epoch % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "cost=", 
#                "{:.9f}".format(avg_cost))
#    print(sess.run([weight,bias]))
#                print(avg_cost)
test_prediction = sess.run(logits, feed_dict={x:train_data})
print(test_prediction)
#
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, "float")))
#print("Accuracy:", accuracy.eval({x: test_prediction, y: test_target}))
#            
    
print("Optimization Finished!")



#print(len(test_prediction),test_prediction)
t = []
for a in test_prediction:
    if a[0] < 0.5:
        t.append(0)
        
    else:
        t.append(1)
print(len(t),len(list(train_target["Survived"])))    
c=0
for i in range(len(t)):
    if t[i] == list(train_target["Survived"])[i]:
        print(t[i],list(train_target["Survived"])[i])
        c+=1
print(c)        
print(c/len(t) *100)             
sess.close()
