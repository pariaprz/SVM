# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:13:57 2023

@author: matrix
"""

import pickle as pk
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import numpy as np
import pandas as pd

data_set = pk.load(open('dataset.pkl','rb'))


data = data_set['training_data']
label = data_set['training_label']


def optimize_Gamma(data, label, C, *gamma):
    for idx, i in enumerate(gamma):
        model = svm.SVC(kernel='rbf', C=C, gamma=i)
        model.fit(data, label)
        h = 0.2
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
       
        colors = ['red' if label == 1 else 'blue' for label in label]
       
        plt.subplot(2,3,idx+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
       
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
       
        Z = Z.reshape(xx.shape)
       
        plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm,alpha=0.8)
       
        plt.scatter(data[:, 0], data[:, 1], c=colors)
  
        plt.xticks(())
        plt.yticks(())
       
        plt.title('gamma = ' + str(i))
        
        pred = model.predict(data)
        
        print('\ngamma:',gamma[idx])
        acc = metrics.accuracy_score(label,pred)
        print('Accuracy: ', acc)

        confusion_matrix = metrics.confusion_matrix(label,pred)
        cm_tabel = pd.DataFrame(confusion_matrix,index = ['P','N'],columns=['P','N'])
        print(cm_tabel)
   
    plt.show()
    
    

def optimize_C(data, label, gamma, *C):
    for idx, i in enumerate(C):
        model = svm.SVC(kernel='rbf', C=i, gamma=gamma)
        model.fit(data, label)
        h = 0.2
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
       
        colors = ['red' if label == 1 else 'blue' for label in label]
       
        plt.subplot(2,3,idx+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
       
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
       
        Z = Z.reshape(xx.shape)
       
        plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm,alpha=0.8)
       
        plt.scatter(data[:, 0], data[:, 1], c=colors)
        
        plt.xticks(())
        plt.yticks(())
       
        plt.title('c = ' + str(i))
        
        pred = model.predict(data)
        
        print('\nC:',C[idx])
        acc = metrics.accuracy_score(label,pred)
        print('Accuracy: ', acc)

        confusion_matrix = metrics.confusion_matrix(label,pred)
        cm_tabel = pd.DataFrame(confusion_matrix,index = ['P','N'],columns=['P','N'])
        print(cm_tabel)
   
    plt.show()
    
   


optimize_C(data,label,0.1,1,10,100,1000,10000)
optimize_Gamma(data,label,1,0.01,1,10)