import tensorflow as tf
import numpy as np
import cifar10_in as inputs
batch_size=50000
images=inputs.get_images()
images=images[range(0,100)]
labels=inputs.get_labels()
labels=labels[range(0,100)]
#images,labels load
test_rate=0.2
hits=0
#print(labels.shape)
sepind=np.floor(labels.shape[0]*test_rate) #first test_rate*100 percent of the images used for test
						#the reminingfor calculate the nearest neighbor  
sepind=sepind.astype(int);
for i in range(0,sepind):
	min_val=np.sum(np.abs(np.divide(images[sepind+1],images[i]))) #diff
	min_label=labels[sepind+1:]
	for j in range(sepind+1,labels.shape[0]):#minimum differance searching
		val=np.sum(np.abs(np.divide(images[j],images[i])))
		if val<min_val:
			min_val=val
			min_label=labels[j]
	if np.argmax(min_label)==np.argmax(labels[i]):
		hits=hits+1  #if the nearest neighbor has the same label
precision=np.divide(hits*100,sepind)
print("Precision of nearest neighbor: "+str(precision)+"%")