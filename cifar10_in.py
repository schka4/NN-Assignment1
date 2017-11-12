#import part
import tensorflow as tf

import numpy as np
import cifar10_input as cifar10

#input data
batch_size=10000
input_data=cifar10.inputs(True, "/home/schka/Documents/deep_learn_sajat/cifar-10-batches-py", batch_size)
init=tf.global_variables_initializer()
sess = tf.Session()
tf.train.start_queue_runners(sess);
sess.run(init)
dec_labels=sess.run(input_data[1])
#print(dec_labels.dtype)
#print(dec_labels.shape)
#dec_labels=dec_labels.tolist()
labels=np.zeros((batch_size,10))
for i in range(0,batch_size-1):
	ind=dec_labels[i]
	labels[i,ind]=1.
images=sess.run(input_data[0])
def get_images():
	return images
def get_labels():
	return labels