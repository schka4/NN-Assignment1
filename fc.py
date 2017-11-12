#import part, inputs
import tensorflow as tf
import cifar10_in as inputs
batch_size=10000
import numpy as np
import random as rnd
imagesm=inputs.get_images()
labels=inputs.get_labels()
images=np.zeros((batch_size, imagesm.shape[1]*imagesm.shape[2]*imagesm.shape[3]))
for i in range(0,batch_size):
	images[i]=np.reshape(imagesm[i],(1,imagesm.shape[1]*imagesm.shape[2]*imagesm.shape[3]))

#parameters, placeholders
tf.reset_default_graph()
train_rate=0.8
steps_in_batch=60
learning_rate=0.005
Batch_num=1500
im_size=images.shape[1:]
im_size=np.asarray(im_size)
#print(im_size)
lab_size=labels.shape[1:]
lab_size=np.asarray(lab_size)
#print(lab_size)
InputData=tf.placeholder(tf.float32,[None,im_size])
DesiredOutput=tf.placeholder(tf.float32,[None,lab_size])

#building up the neural network
elso=im_size[0]
utolso=lab_size[0]
neurnos_by_layers=[elso,1024,1024,utolso]
Input=InputData
for i in range(len(neurnos_by_layers)-1):
  with tf.variable_scope('layer' +str(i)):
    W=tf.Variable(tf.random_normal([neurnos_by_layers[i],neurnos_by_layers[i+1]]))
    b=tf.Variable(tf.random_normal([neurnos_by_layers[i+1]]))
    Input=tf.add(tf.matmul(Input,W),b)
    if i!=(len(neurnos_by_layers)-2):
      Input=tf.sigmoid(Input)
Output=Input

with tf.name_scope('loss'):
  loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(DesiredOutput,Output)  )
  #loss=tf.reduce_mean(tf.abs((tf.subtract(Output,DesiredOutput))))
with tf.name_scope('optimizer'):
  optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.name_scope('precision'):
  hit=tf.equal(tf.argmax(Output,1),tf.argmax(DesiredOutput,1))
  precision=tf.reduce_sum(tf.cast(hit,tf.float32))

#Training
init=tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  end_ind=np.dtype('int64').type(np.floor(batch_size*train_rate))
  Indices=np.asarray(range(0,end_ind)) #elso 80 szazalekkal tanitjuk-> lehetseges indexek
  #print(Indices.shape)
  for i in range(0,Batch_num-1):
    Batch_ind =Indices[rnd.sample( range(0,Indices.shape[0]) ,steps_in_batch)] #kivalaszt steps_in_batch db-ot
    X=images[Batch_ind,:]
    Yd=labels[Batch_ind,:]
    _,l=sess.run([optimizer,loss], feed_dict={InputData:X , DesiredOutput: Yd})
    if (i%500)==0:
      print('loss: ' +str(l))
  #Testing
  starting_index=end_ind+1
  X_test=images[starting_index:]
  Y_test=labels[starting_index:]
  
  p = sess.run(precision, feed_dict={InputData: X_test, DesiredOutput: Y_test})
  #print("Num of hits: "+str(p))
  prec=float(p)*100/X_test.shape[0]
  print("Precision of fully connected: "+str(float(p)*100/X_test.shape[0])+" %")

