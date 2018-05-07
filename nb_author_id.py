#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import pandas as pd
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


features_trainDF = pd.DataFrame(features_train)
features_testDF = pd.DataFrame(features_test)

labels_trainDF = pd.DataFrame(labels_train)
labels_testDF = pd.DataFrame(labels_test)

#print features_train[15804, 3:10]
#print features_trainDF
features_trainDF.to_csv("./files/features_train.csv", index_label= False, header = False)
features_testDF.to_csv("./files/features_test.csv", index_label= False, header =False)

labels_testDF.to_csv("./files/labels_test.csv", index_label= False, header=False)
labels_trainDF.to_csv("./files/labels_train.csv", index_label=False, header=False)

#########################################################
### your code goes here ###

from sklearn.naive_bayes import  GaussianNB

t0 = time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"



t0 = time()
#pred = clf.predict(features_test)
print "pred time:", round(time()-t0, 3), "s"

print float(sum(pred == labels_test))/len(labels_test)

#########################################################


