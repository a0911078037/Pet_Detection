# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:19:47 2022

@author: xvideo
"""
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from data import data
from keras.models import load_model


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(5,5), padding='same',input_shape=(40,40,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=20, kernel_size=(5,5), padding='same',input_shape=(40,40,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def preprocess_data(data):
    data.train_feature_vector = data.train_feature.reshape(len(data.train_feature), 40,40,3).astype('float32')
    data.test_feature_vector = data.test_feature.reshape(len(data.test_feature), 40,40,3).astype('float32')
    data.train_feature_normalize = data.train_feature_vector / 255
    data.test_feature_normailize = data.test_feature_vector / 255
    data.train_label_onehot = np_utils.to_categorical(data.train_label)
    data.test_label_onehot = np_utils.to_categorical(data.test_label)
    return data

def show_images_labels_predictions(images,labels,predictions,start_id,num=10):
    label_dicts =['Cat','Dog']
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示黑白圖片
        ax.imshow(images[start_id])
        
    # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + label_dicts[predictions[start_id]]
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if (predictions[start_id]==labels[start_id]) else ' (x)') 
            title += '\nlabel = ' + label_dicts[(labels[start_id])]
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[start_id])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
        plt.show()
if __name__ == '__main__':
    path = 'imageData/'
    data = data()
    data.train_feature = np.load(path+'train_feature.npy')
    data.train_label = np.load(path+'train_label.npy')
    data.test_feature = np.load(path+'test_feature.npy')
    data.test_label = np.load(path+'test_label.npy')
    data = preprocess_data(data)
    model =  None
    model = load_model('detect_animals.h5')
    if model is None:
        print('cannot load model')
        model = create_model()
        train_histroy = model.fit(x=data.train_feature_normalize, y=data.train_label_onehot,
                              validation_split=0.2,epochs=10,batch_size=200,verbose=2)
    predictions = model.predict(data.test_feature_normailize)
    predictions = np.argmax(predictions, axis=1)
    del model
    show_images_labels_predictions(data.test_feature, data.test_label, predictions, 0)
     
    