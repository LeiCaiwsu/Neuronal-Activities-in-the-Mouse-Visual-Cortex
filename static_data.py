# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:55:25 2016

@author: lei.cai
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import math
from sklearn.svm import SVC
import parameter_select as ps


def generate_label(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][1]) or math.isnan(data[index][2]):
            label[index]=-1
        else:
            label[index]=20*(data[index][0]/30)+4*(math.log(int(round(data[index][1],2)/0.02),2))+ data[index][2]/0.25
    return label
    
def generate_label_orientation(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][1]) or math.isnan(data[index][2]):
            label[index]=-1
        else:
            label[index]=data[index][0]/30
    return label
    
def generate_label_orientation_nospatial(data,spatial):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][1]) or math.isnan(data[index][2]) or round(data[index][1],2)!=spatial:
            label[index]=-1
        else:
            label[index]=data[index][0]/30
    return label
    
def generate_label_spatial(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][1]) or math.isnan(data[index][2]):
            label[index]=-1
        else:
            label[index]= math.log(int(round(data[index][1],2)/0.02),2)
    return label
    
def generate_label_spatial_noori(data,ori):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][1]) or math.isnan(data[index][2]) or data[index][0]/30!=ori:
            label[index]=-1
        else:
            label[index]= math.log(int(round(data[index][1],2)/0.02),2)
    return label
    
def normalize(data):
    min_idx = np.argmin(data,axis=0)
    max_idx = np.argmax(data,axis=0)
    min_value = data[min_idx,np.arange(data.shape[1])]
    max_value = data[max_idx,np.arange(data.shape[1])]
    return (data-min_value)/(max_value-min_value), min_value,max_value
    

boc = BrainObservatoryCache(manifest_file='manifest.json')

ophys = boc.get_ophys_experiments()

sessionB = [x['id'] for x in ophys if x['session_type']=='three_session_B' ]

##################################

acc = []
struct = []
depth = []
cre_line = []
type = []
i = 0
dim_fea = []
score_linear = []
score_rbf = []
train_his = []
Num_aver = 30
test_his = []


for id in sessionB:
    print "processing experiment: ",id," the ",i,"th of ",len(sessionB)," ."
    data = boc.get_ophys_experiment_data(id)

    metadata = data.get_metadata()
    struct.append(metadata['targeted_structure'])
    depth.append(metadata['imaging_depth_um'])
    cre_line.append(metadata['cre_line'])
    type.append(metadata['genotype'])

    static = data.get_stimulus_table(data.list_stimuli()[3])
    static_value = static.values


    dff_data = data.get_dff_traces()
    dff_value = dff_data[1]

    ### select one label to analysis the activity  ######

    #label = generate_label_spatial_noori(static_value[:,0:3])
    label = generate_label_spatial(static_value[:,0:3])
    #label = generate_label_orientation_nospatial(static_value[:,0:3],0.02)
    #label = generate_label_orientation(static_value[:,0:3])
    
    label = label.astype(int)


    feature = ps.extract(dff_value,static_value,1)    

    feature,label = ps.data_filter(feature,label)

    ##############using svm to analysis the activity#########
    ac_aver = 0
    for num in range(Num_aver):
        trainidx,testidx = ps.split_train_test(label,0.8)
        train_his.append(trainidx)
        test_his.append(testidx)
        ac_aver += ps.svm_clf(feature[trainidx],label[trainidx],feature[testidx],label[testidx])    
    acc.append(ac_aver/Num_aver)
    np.savez("./spatial_fea/feature_"+str(id),feature,label)

    i+=1
    
    
   
acc_struct_depth =[]
for area in np.unique(struct):
    for dpt in np.unique(depth):
        select = [x for x in np.arange(len(sessionB)) if struct[x]==area and depth[x]==dpt]
        acc_struct_depth.append(np.average(np.array(acc)[select]))