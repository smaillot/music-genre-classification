from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import math 
import sklearn
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf  
from sklearn.decomposition import PCA

tf.logging.set_verbosity(tf.logging.INFO)

# In this section we take the  original array of sample and label and take the PCA

################################################################
features = np.load('normalized.npy')
labels=np.load('labels.npy')

# ## PCA

pca = PCA()
pca.fit(features)
cumvar = np.cumsum(pca.explained_variance_ratio_)

##########################
## choose variance to keep
var_th = 0.5
##########################
for i in range(len(cumvar)):
    if cumvar[i] >= var_th:
        n_comp = i
        break
print('n_components to keep {}% of the variance : {}'.format(var_th*100, n_comp))
X_pca = pca.transform(features)[:,:n_comp]
print('shape of dim reducted data : ' + str(np.shape(X_pca)))


def get_PCA(X):
    return pca.transform(X)[:,:55]

features = get_PCA(features)


# Replace the name of the music genre by their label
def associate_label(label,name_Type):
    labels=[]
    for k in label:
        for j in range(10):
            if k==name_Type[j]:
                labels.append(j)
    return labels

# Separate data and labels
def separation(dataset,name_Type):
    list_data=[]
    for k in range(10):
        data=[[dataset[l,:]] for l in range(len(dataset)) if dataset[l,-1]==k]
        list_data.append(data)
    return list_data

# Internally shuffle each music genre 
def internal_shuffle(list_dataset):
    list_train=[]
    list_test=[]
    for k in range(10):
        working_Array=np.reshape(np.array(list_dataset[k]),[100,56])
        np.random.shuffle(working_Array)
        list_train.append(working_Array[:25,:])
        list_test.append(working_Array[25:,:])
    return list_train, list_test

# if j=0 concatenate all the samples from each music genre for the training set
# if j=1 concatenate all the samples from each music genre for the training set
# then it shuffle
def concatenate_matrix(data,j,wanted):
    if j==0:
        matrix=np.reshape(np.array(data[wanted[0]]),[25,56])
        for k in range(wanted[0]+1,10):
            if k in wanted:
                matrix=np.concatenate((matrix,np.reshape(np.array(data[k]),[25,56])))
    elif j==1:
        matrix=np.reshape(np.array(data[wanted[0]]),[75,56])
        for k in range(wanted[0]+1,10):
            if k in wanted:
                matrix=np.concatenate((matrix,np.reshape(np.array(data[k]),[75,56])))
    np.random.shuffle(matrix)
    return matrix

def label_preparation(data):
    dataset=data[:,:55]
    lab=data[:,-1]
    labb=[int(l) for l in lab]
    return dataset, labb

# Labels associated to each type of music
name_Type=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
# Labels      0        1          2         3       4        5     6        7      8       9

wanted=[0,1,2,3,4,5,6,7,8,9]

for test in range(1000,1010,1):
 # initialization of the mean values
 meanTP=0
 meanTN=0
 meanFP=0
 meanFN=0
 Accuracy=0
 error=0
 for a in range(1,2):
  lab=np.reshape(associate_label(labels,name_Type),[1000,1])
  features_And_Label=np.concatenate((features,lab),axis=1)

 # Put in a list of matrices of samples regrouped by type
  list_Of_Data=separation(features_And_Label,name_Type)

 # Shuffle the matrices of sample and take 80% for training and 20% for testing
  [list_Training_Samples,list_Testing_Samples]=internal_shuffle(list_Of_Data)

 # Generate a matrix with randomly shuffled samples for training and testing
  training_Samples=concatenate_matrix(list_Training_Samples,0,wanted)
  testing_Samples=concatenate_matrix(list_Testing_Samples,1,wanted)

  [training_Data,training_Labels]=label_preparation(training_Samples)
  [testing_Data,testing_Labels]=label_preparation(testing_Samples)

  l=150
  m=150
  n=150
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=55)]
  classifier = tf.contrib.learn.DNNClassifier(hidden_units=[l, m, n],
      feature_columns=feature_columns,
      model_dir="/tmp/model"+str(l)+str(m)+str(n)+str(test),
      n_classes=10,
      weight_column_name=None,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
      activation_fn=tf.nn.softplus)
#    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    # Define the training inputs
  def get_train_inputs():
      x = tf.constant(training_Data)
      y = tf.constant(training_Labels)
      return x, y
# Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=6000)#,monitors=[validation_monitor]
  def get_test_inputs():
      x = tf.constant(testing_Data)
      y = tf.constant(testing_Labels)
    
      return x, y
    
# Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
  predict_tf = classifier.predict(input_fn = get_test_inputs)


  prediction=[i for i in predict_tf]

  cnf_matrix = confusion_matrix(testing_Labels, prediction)
  num_element=cnf_matrix[0][0]+cnf_matrix[0][1]+cnf_matrix[1][0]+cnf_matrix[1][1]
  meanTP+=cnf_matrix[1][1]/(num_element)
  meanTN+=cnf_matrix[0][0]/(num_element)
  meanFP+=cnf_matrix[0][1]/(num_element)
  meanFN+=cnf_matrix[1][0]/(num_element)
  Accuracy+=accuracy_score
  error+=(cnf_matrix[0][1]+cnf_matrix[1][0])/(num_element)
  
  with open("global.txt", "a") as myfile:
       myfile.write(str(m)+","+str(m)+","+str(m)+","+str(Accuracy)+"\n")


# to visualize tensorboard tap in the terminal:
#    tensorboard --logdir=path/to/log-directory