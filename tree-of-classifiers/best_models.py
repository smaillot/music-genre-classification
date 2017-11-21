__author__      = "tdesquins"

# Tree of classifier
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
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


features = np.load('normalized.npy')
labels=np.load('labels.npy')

# ## PCA

pca = PCA()
pca.fit(features)
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumvar)
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
plt.plot(n_comp, var_th, 'or', markersize=6)

def get_PCA(X):
    return pca.transform(X)[:,:55]

features = get_PCA(features)



tf.logging.set_verbosity(tf.logging.INFO)

def confusion_tot(cm):
    cm = np.array([list(cm[i,:]) + [np.sum(cm[i,:])] for i in range(np.shape(cm)[0])])
    last_line = [np.sum(cm[:,i]) for i in range(np.shape(cm)[1] - 1)]
    last_line = last_line + [np.sum(last_line)]
    cm = np.concatenate((cm,np.array([last_line])), axis=0)
    return cm

def confusion_prob(cm):
    prob = np.zeros(np.shape(cm))
    prob[:-1,:-1] = 100 * cm[:-1,:-1] / np.sum(cm[:-1,:-1])
    prob[:-1,-1] = 100 * np.diag(cm[:-1,:-1]) / np.sum(cm[:-1,:-1], axis=1)
    prob[-1,:-1] = 100 * np.diag(cm[:-1,:-1]) / np.sum(cm[:-1,:-1], axis=0)
    prob[-1,-1] = 100 * np.sum(np.diag(cm[:-1,:-1])) / np.sum(cm[:-1,:-1])
    return prob

def plot_confusion_matrix(y, y_pred, title='Confusion matrix', cmap=plt.cm.Oranges, display=True):
    dim = max(len(np.unique(y)), len(np.unique(y_pred)))
    cm = confusion_matrix(y, y_pred)
    cm_red = np.zeros((len(cm)+1,len(cm)+1))
    cm_red[:-1,:-1] = confusion_prob(cm)
    cm = confusion_tot(cm)
    cm_prob = confusion_prob(cm)
    if display:
        plt.figure(figsize=(15,1.5*(dim+1)))
        plt.imshow(cm_prob, interpolation='nearest', cmap=cmap)
        plt.title(title + ' - accuracy : {:.2f}%'.format(100 * np.sum(np.diag(cm)) / np.sum(cm)))
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, list(np.unique(y)) + ['Total'], rotation=45)
        plt.yticks(tick_marks, list(np.unique(y)) + ['Total'])
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:.0f}\n({:.2f}%)'.format(cm[i, j],cm_prob[i, j]), horizontalalignment="center", size=12, color="green" if i == j or i+1 == len(cm) or j+1 == len(cm) else "red")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    if not display:
        return 100 * np.sum(np.diag(cm))/ np.sum(cm)

def associate_label(label,name_Type):
    labels=[]
    for k in label:
        for j in range(10):
            if k==name_Type[j]:
                labels.append(j)
    return labels

def separation(dataset,name_Type):
    list_data=[]
    for k in range(10):
        data=[[dataset[l,:]] for l in range(len(dataset)) if dataset[l,-1]==k]
        list_data.append(data)
    return list_data

def internal_shuffle(list_dataset):
    list_train=[]
    list_test=[]
    for k in range(10):
        working_Array=np.reshape(np.array(list_dataset[k]),[100,56])
        np.random.shuffle(working_Array)
        list_train.append(working_Array[:25,:])
        list_test.append(working_Array[25:,:])
    return list_train, list_test

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
    list_lab=[]
    for k in range(10):
        list_lab.append([int(la==k) for la in lab])
    return dataset, list_lab

# name of the types of music
name_Type=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
# Labels      0        1          2         3       4        5     6        7      8       9
a=15
# Choice of the label We want to train
wanted=[0,1,2,3,4,5,6,7,8,9]
order=[4,5,1,2,3,0,6,7,8,9]
all_Training_Data=[]
all_Labeltrain_Data=[]
all_Testing_Data=[]
all_Labeltest_Data=[]

lab=np.reshape(associate_label(labels,name_Type),[1000,1])
features_And_Label=np.concatenate((features,lab),axis=1)

# Put in a list of matrices of samples regrouped by type
list_Of_Data=separation(features_And_Label,name_Type)

# Shuffle the matrices of sample and take 25% for training and 75% for testing
[list_Training_Samples,list_Testing_Samples]=internal_shuffle(list_Of_Data)

# We use the same functions as before but by automatically repeating the operation 
# with different wanted vectors according to the branch we wante to train, without 
# forgetting that the last class , Rock has to be trained with all the music genre
for j in range(10):
 if j==0:

  # Generate a matrix with randomly shuffled samples for training and testing
  training_Samples=concatenate_matrix(list_Training_Samples,0,wanted)
  testing_Samples=concatenate_matrix(list_Testing_Samples,1,wanted)

  # Generate the lists of label used to train of test the classifier
  [training_Data,training_Labels]=label_preparation(training_Samples)
  [testing_Data,testing_Labels]=label_preparation(testing_Samples)
  all_Training_Data.append(training_Data)
  all_Labeltrain_Data.append(training_Labels[4])
  anend=training_Labels[9]
 else:
  if j!=9:
   wanted=[l for l in wanted if l!=order[j-1] ]
  # Generate a matrix with randomly shuffled samples for training and testing
   training_Samples=concatenate_matrix(list_Training_Samples,0,wanted)

  # Generate the lists of label used to train of test the classifier
   [training_Data,training_Labels]=label_preparation(training_Samples)
   all_Training_Data.append(training_Data)
   all_Labeltrain_Data.append(training_Labels[order[j]])
  else:
     all_Training_Data.append(all_Training_Data[0])
     all_Labeltrain_Data.append(anend) 


 # to precise that our features have real values
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=55)]



 ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
 # Definition of the classifier
 
classifier0 = tf.contrib.learn.DNNClassifier(hidden_units=[100, 130, 100],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(100)+str(130)+str(100)+str(4)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)

classifier1 = tf.contrib.learn.DNNClassifier(hidden_units=[80, 120, 130],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(80)+str(120)+str(130)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier2 = tf.contrib.learn.DNNClassifier(hidden_units=[60, 70, 50],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(60)+str(70)+str(50)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier3 = tf.contrib.learn.DNNClassifier(hidden_units=[120, 110, 110],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(120)+str(110)+str(110)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier4 = tf.contrib.learn.DNNClassifier(hidden_units=[170, 140, 110],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(170)+str(140)+str(110)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier5 = tf.contrib.learn.DNNClassifier(hidden_units=[105, 115, 110],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(105)+str(115)+str(110)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier6 = tf.contrib.learn.DNNClassifier(hidden_units=[57, 53, 47],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(57)+str(53)+str(47)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier7 = tf.contrib.learn.DNNClassifier(hidden_units=[70, 70, 60],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(70)+str(70)+str(60)+str(j)+str(100)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)


classifier8 = tf.contrib.learn.DNNClassifier(hidden_units=[40, 50, 60],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(40)+str(50)+str(60)+str(j)+str(1000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)

classifier9 = tf.contrib.learn.DNNClassifier(hidden_units=[50, 100, 100],
    feature_columns=feature_columns,
    model_dir="./tmp/mode"+str(40)+str(50)+str(60)+str(j)+str(10000)+str(a),
    n_classes=2,
    weight_column_name=None,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
    activation_fn=tf.nn.softplus)

train_Labels=all_Labeltrain_Data[0]
train=all_Training_Data[0]


 # Train all the classifiers

def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y


classifier0.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[1]
train=all_Training_Data[1]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier1.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[2]
train=all_Training_Data[2]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier2.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[3]
train=all_Training_Data[3]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier3.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[4]
train=all_Training_Data[4]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier4.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[5]
train=all_Training_Data[5]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier5.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[6]
train=all_Training_Data[6]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier6.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[7]
train=all_Training_Data[7]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier7.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[8]
train=all_Training_Data[8]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier8.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]
train_Labels=all_Labeltrain_Data[9]
train=all_Training_Data[9]
def get_train_inputs():
    x=tf.constant(train)
    y = tf.constant(train_Labels)
    return x, y
classifier9.fit(input_fn=get_train_inputs, steps=5000)#,monitors=[validation_monitor]


 
error=0;

# test through all the branch of the tree
for j in range(10):

  test_Labels=testing_Labels[order[j]]
  
  def get_test_inputs():
      x = tf.constant(testing_Data)
      y = tf.constant(test_Labels)
      return x, y

  def get_test_inputs2():
      x = tf.constant(testing_Data)
      return x
 


# Predict according to the classifier of the branch
  if j==0:
      predict_tf = classifier0.predict(input_fn = get_test_inputs2)
  elif j==1:
      predict_tf = classifier1.predict(input_fn = get_test_inputs2)
  elif j==2:
      predict_tf = classifier2.predict(input_fn = get_test_inputs2)
  elif j==3:
      predict_tf = classifier3.predict(input_fn = get_test_inputs2)
  elif j==4:
      predict_tf = classifier4.predict(input_fn = get_test_inputs2)
  elif j==5:
      predict_tf = classifier5.predict(input_fn = get_test_inputs2)
  elif j==6:
      predict_tf = classifier6.predict(input_fn = get_test_inputs2)
  elif j==7:
      predict_tf = classifier7.predict(input_fn = get_test_inputs2)
  elif j==8:
      predict_tf = classifier8.predict(input_fn = get_test_inputs2)
  elif j==9:
      predict_tf = classifier9.predict(input_fn = get_test_inputs2)
 

  prediction=[i for i in predict_tf]
  print(prediction)
  cnf_matrix = confusion_matrix(test_Labels, prediction)
  next1=[ ]
  la0=[]
  la1=[]
  la2=[]
  la3=[]
  la4=[]
  la5=[]
  la6=[]
  la7=[]
  la8=[]
  la9=[]
  
  # Generate the new labels and new samples by those which has been classified as 0
  # in the last branch
  for k in range(len(prediction)):
      if prediction[k]==0:
          next1.append(testing_Data[k,:])
          la0.append(testing_Labels[0][k])
          la1.append(testing_Labels[1][k])
          la2.append(testing_Labels[2][k])
          la3.append(testing_Labels[3][k])
          la4.append(testing_Labels[4][k])
          la5.append(testing_Labels[5][k])
          la6.append(testing_Labels[6][k])
          la7.append(testing_Labels[7][k])
          la8.append(testing_Labels[8][k])
          la9.append(testing_Labels[9][k])
  print(testing_Labels[order[1]])
           
  testing_Labels=[la0,la1,la2,la3,la4,la5,la6,la7,la8,la9]
  testing_Data=np.vstack(next1)
   

  
#  plot_confusion_matrix(test_Labels, prediction, title='Confusion matrix', cmap=plt.cm.Oranges, display=True)
  num_element=cnf_matrix[0][0]+cnf_matrix[0][1]+cnf_matrix[1][0]+cnf_matrix[1][1]
  accu=(cnf_matrix[0][0]+cnf_matrix[1][1])/num_element
  with open("final_tree_result.txt", "a") as myfile:
    myfile.write("\n"+str(cnf_matrix)+" ,")
  with open("final_tree_resultexcel.txt", "a") as myfile2:
    myfile2.write("\n"+str(cnf_matrix[1][1]/num_element*100)+"%"+" ,"+str(cnf_matrix[0][0]/num_element*100)+"%"+" ,"+str(cnf_matrix[0][1]/num_element*100)+"%"+", "+str(cnf_matrix[1][0]/num_element*100)+" ,"+str(accu)+" ,"+str(cnf_matrix[1][1]/750*100)+"%"+" ,"+str(cnf_matrix[0][0]/750*100)+" ,"+str(cnf_matrix[0][1]/750*100)+" ,"+str(cnf_matrix[1][0]/750*100)+"%"+" ,"+str(num_element))
      #
 ## to visualize tensorboard tap in the terminal:
 ##    tensorboard --logdir=path/to/log-directory