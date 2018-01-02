# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:20:32 2016

@author: lei.cai
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import math
from sklearn import svm
import parameter_select as ps

def generate_label(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]):
            label[index]=-1
        else:
            label[index]= 8*math.ceil(math.log(data[index][0],2))+int(data[index][1]/45)
    return label
    
def generate_fre(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]):
            label[index]=-1
        else:
            label[index]=math.ceil(math.log(data[index][0],2))
    return label

def generate_fre_givenori(data,ori):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]) or int(data[index][1]/45)!=ori:
            label[index]=-1
        else:
            label[index]=math.ceil(math.log(data[index][0],2))
    return label

def generate_ori(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]):
            label[index]=-1
        else:
            label[index]=int(data[index][1]/45)
    return label
    
def generate_ori_givenfre(data,fre):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]) or math.ceil(math.log(data[index][0],2))!=fre:
            label[index]=-1
        else:
            label[index]=int(data[index][1]/45)
    return label
    
def generate_nosense_ori(data):
    label = np.zeros(data.shape[0])
    for index in np.arange(data.shape[0]):
        if math.isnan(data[index][0]) or math.isnan(data[index][0]):
            label[index]=-1
        else:
            if data[index][1]>179:
                label[index] = int((data[index][1]-180)/45)
            else:
                label[index]=int(data[index][1]/45)
    return label

boc = BrainObservatoryCache(manifest_file='manifest.json')
ophys = boc.get_ophys_experiments()
sessionA = [x['id'] for x in ophys if x['session_type']=='three_session_A']

time=0
acc = []
struct = []
depth = []
cre_line = []
extype = []
Num_aver = 30
train_his = []
test_his = []
plabel_tot = []
tlabel_tot = []
ac_tot = []

for id in sessionA:
    print "processing drift experiment: ",id," the ",time,"th of ",len(sessionA)," ."
    data = boc.get_ophys_experiment_data(id)
    metadata =  data.get_metadata()
    struct.append(metadata['targeted_structure'])
    depth.append(metadata['imaging_depth_um'])
    cre_line.append(metadata['cre_line'])
    extype.append(metadata['genotype'])
    drifting = data.get_stimulus_table(data.list_stimuli()[0])
    drift = drifting.values
    
    dff_data = data.get_dff_traces()
    dff_value = dff_data[1]

    ### select one label to analysis the activity  ######
    
    #label = generate_label(drift[:,0:2])
    #label = generate_fre(drift[:,0:2])
    label = generate_fre_givenori(drift[:,0:2],7)   #0-7
    #label = generate_ori(drift[:,0:2])
    #label = generate_ori_givenfre(drift[:,0:2],5)  #0-4

    ######################################################


    label = label.astype(int)

    feature = ps.extract(dff_value,drift,0)    
    
    feature,label = ps.data_filter(feature,label)
    
    ac = []
    plabel = []
    tlabel = []
    for num in range(Num_aver):
        trainidx,testidx = ps.split_train_test(label,0.80)
        train_his.append(trainidx)
        test_his.append(testidx)
        lin_clf = svm.LinearSVC(C=1)
        lin_clf.fit(feature[trainidx],label[trainidx])
        ac.append(lin_clf.score(feature[testidx],label[testidx]))
        pre_label = lin_clf.predict(feature[testidx])
        true_label = label[testidx]
        plabel.append(pre_label)
        tlabel.append(true_label)
    plabel_tot.append(plabel)
    tlabel_tot.append(tlabel)
    ac_tot.append(ac)
    #np.savez("./drift/fre_ori=7/feature_"+str(id),feature,label)
    time+=1

#np.savez("./drift/result_ac_fre_ori_7.npz",ac_tot,train_his,test_his,plabel_tot,tlabel_tot)    


# get average accuracy based on different area and depth
acc = np.average(ac_tot,axis=1)
acc_struct_depth =[]
for area in np.unique(struct):
    for dpt in np.unique(depth):
        select = [x for x in np.arange(len(sessionA)) if struct[x]==area and depth[x]==dpt]
        acc_struct_depth.append(np.average(np.array(acc)[select]))