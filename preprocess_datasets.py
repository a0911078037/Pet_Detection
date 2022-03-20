# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:50:00 2022

@author: xvideo
"""
import os,cv2,glob
from sklearn.model_selection import train_test_split
import numpy as np

images = []
labels = []
dict_labels = {'Cat':0, 'Dog':1}
size = (40, 40)

#loading data
for folders in glob.glob('imageData/*'):
    for filename in os.listdir(folders):
        label = folders.split("\\")[-1]
        try:
            img = cv2.imread(os.path.join(folders,filename))
            if img is not None:
                img = cv2.resize(img, dsize=size)
                images.append(img)
                labels.append(dict_labels[label])
        except:
            print(os.path.join(folders,filename),'cannot read!')
            pass
#preprcess data
train_feature, test_feature, train_label, test_label = \
    train_test_split(images, labels, test_size=0.2, random_state=0)
    
train_feature = np.array(train_feature)
test_feature = np.array(test_feature)
train_label = np.array(train_label)
test_label = np.array(test_label)
path = 'imageData/'
np.save(path+'train_feature.npy',train_feature)
np.save(path+'test_feature.npy',test_feature)
np.save(path+'train_label.npy',train_label)
np.save(path+'test_label.npy',test_label)