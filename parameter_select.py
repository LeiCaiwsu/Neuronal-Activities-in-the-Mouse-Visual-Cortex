# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:18:56 2016

@author: lei.cai
"""

from sklearn import svm
import numpy as np
import random

def test_lin_clf(traindata,label):
    num_group = 5
    train_fold = np.array_split(traindata,num_group)
    label_fold = np.array_split(label,num_group)
    c_choices = [0.01,1,10,100,1000]
    score_linear_c = []
    score_rbf_c = []
    for c in c_choices:
        
        for i in range(num_group):
            X_train = np.vstack(train_fold[0:i]+train_fold[i+1:])
            X_test = train_fold[i]
            Y_train = np.hstack(label_fold[0:i]+label_fold[i+1:])
            Y_test = label_fold[i]
            lin_clf = svm.LinearSVC(C=c)
            lin_clf.fit(X_train,Y_train)
            score = lin_clf.score(X_test,Y_test)
            score_linear_c.append(score)
            
            rbf_clf = svm.SVC(C=c)
            rbf_clf.fit(X_train,Y_train)
            score_rbf_c.append(rbf_clf.score(X_test,Y_test))
    
    print "score for linear:",score_linear_c
    print "score for rbf:",score_rbf_c
    return score_linear_c,score_rbf_c
    
def svm_clf(traindata,trainlabel,testdata,testlabel):
    lin_clf = svm.LinearSVC(C=1)
    lin_clf.fit(traindata,trainlabel)
    return lin_clf.score(testdata,testlabel)
    
def test_svm_numtrain(data,label):
    N = data.shape[0]
    PerofTrain = [0.2,0.4,0.6,0.8,0.9]
    trainidx = []
    testidx = []
    score = []
    for i in PerofTrain:
        train_num = int(N*i)
        trainidx = np.arange(0,train_num)
        testidx = np.arange(train_num,N)
        lin_clf = svm.LinearSVC(C=1)
        lin_clf.fit(data[trainidx],label[trainidx])
        score.append(lin_clf.score(data[testidx],label[testidx]))
    return score
            
def split_train_test(label,percent):
    nclass = len(np.unique(label))
    N = len(label)
    train_num = int(N/nclass*percent)
    trainidx = []
    testidx = []
    label_count = np.zeros(nclass)
    random_list = random.sample(np.arange(N),N)
    for i in random_list:
        if label_count[label[i]]<train_num:
            label_count[label[i]] += 1
            trainidx.append(i)
        else:
            testidx.append(i)
    return trainidx,testidx
    
def extract(data,duration,mode):
    feature = np.zeros((duration.shape[0],data.shape[0]))
    
    if mode==1: #average
        for i in np.arange(duration.shape[0]):
            feature[i] = np.average(data[:,int(duration[i,3]):int(duration[i,4])+1],axis=1)
    else:
        for i in np.arange(duration.shape[0]):
            feature[i] = np.max(data[:,int(duration[i,3]):int(duration[i,4])+1],axis=1)
#        feature = np.zeros((duration.shape[0],data.shape[0]*lgh))
#        for i in np.arange(duration.shape[0]):
#            dur_len = duration[i,4]-duration[i,3]+1
#            if dur_len == lgh:
#                feature[i] = np.hstack(data[:,int(duration[i,3]):int(duration[i,4])+1].T)
#            else:
#                if (dur_len - lgh)%2 ==0:
#                    bias = int((dur_len - lgh)/2)
#                    feature[i] = np.hstack(data[:,int(duration[i,3])+bias:int(duration[i,4]-bias+1)].T)
#                else:
#                    bias = int((dur_len - lgh)/2)
#                    feature[i] = np.hstack(data[:,int(duration[i,3])+bias:int(duration[i,4])-bias].T)
    return feature
    
def data_filter(feature,label):
    filter_index = []
    for index in np.arange(label.shape[0]):
        if label[index] != -1:
            filter_index.append(index)
        
    feature = feature[filter_index,:]
    label = label[filter_index]
    return feature,label

   
    
    