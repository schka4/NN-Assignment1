#import part, inputs
import tensorflow as tf
import cifar10_in as inputs
batch_size=10000
import numpy as np
import random as rnd
images=inputs.get_images()
labels=inputs.get_labels()

#parameters, placeholders
clacc_num=labels.shape[1]
tf.reset_default_graph()
train_rate=0.8
steps_in_batch=60
learning_rate=0.005
Batch_num=1500
im_size=images.shape[1:]
im_size=np.asarray(im_size).tolist()
#print(im_size)
lab_size=labels.shape[1:]
lab_size=np.asarray(lab_size)
#print(lab_size)
InputData=tf.placeholder(tf.float32,[None] + im_size )
DesiredOutput=tf.placeholder(tf.float32,[None,lab_size])

#building up the neural network
kernels_by_layers=[30,128,52,20]
Input=InputData
PrevFilters=im_size[2]
for i in range(0,len(kernels_by_layers)-1):
  kern_num=kernels_by_layers[i]
  #convolutions
  with tf.variable_scope('conv' +str(i)):
    W =tf.get_variable('W', [3,3,PrevFilters,kern_num])
    b=tf.get_variable('b', [kern_num],initializer=tf.constant_initializer(0.0))
    ConvResult = tf.add(tf.nn.conv2d(Input,W,strides=[1,1,1,1], padding='VALID'),b)
    PrevFilters=kern_num
    # relu
    relu=tf.nn.relu(ConvResult)
    #pooling
    Input=tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

with tf.variable_scope('fully_conn'):
        CurrentShape=Input.get_shape()
        #print(CurrentShape)
        FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
        conv_out = tf.reshape(Input, [-1, FeatureLength])
        #print(conv_out.shape)
        W = tf.get_variable('W',[FeatureLength,clacc_num])
        Bias = tf.get_variable('Bias',[clacc_num])
        Output = tf.add(tf.matmul(conv_out, W), Bias)
#print(Output)

with tf.name_scope('loss'):
  loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(DesiredOutput,Output)  )
  #loss=tf.reduce_mean(tf.abs((tf.subtract(Output,DesiredOutput))))
with tf.name_scope('optimizer'):
  optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.name_scope('precision'):
  hit=tf.equal(tf.argmax(Output,1),tf.argmax(DesiredOutput,1))
  precision=tf.reduce_sum(tf.cast(hit,tf.float32))
  #precision=tf.reduce_mean(tf.cast(hit,tf.float32))


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
  print("Precision of convolutional: "+str(float(p)*100/X_test.shape[0])+" %")