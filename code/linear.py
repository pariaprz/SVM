# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:09:00 2023

@author: matrix
"""

import pickle as pk
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

data_set = pk.load(open('dataset.pkl','rb'))


data = data_set['training_data']
label = data_set['training_label']


plt.scatter(data[label==1][:,0], data[label==1][:,1],color='red',label='label 1')
plt.scatter(data[label==-1][:,0], data[label==-1][:,1],color='blue',label='label -1')

plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.title('Scatter ploting of data')

plt.show()



model = svm.SVC(kernel = 'linear',C=1000)
model.fit(data,label)

h = 0.2
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))


colors = ['red' if label == 1 else 'blue' for label in label]


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    # Plot also the training points
plt.scatter(data[:, 0], data[:, 1], c=colors)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xticks(())
plt.yticks(())
plt.title('SVM with linear kernel')
plt.show()


pred = model.predict(data)

acc = metrics.accuracy_score(label,pred)
print('\nAccuracy: ', acc,'\n')

confusion_matrix = metrics.confusion_matrix(label,pred)
cm_tabel = pd.DataFrame(confusion_matrix,index = ['P','N'],columns=['P','N'])
print(cm_tabel)