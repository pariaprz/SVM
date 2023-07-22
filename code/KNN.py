# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:26:00 2023

@author: matrix
"""
import pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk
from sklearn.model_selection import cross_val_score

data_set = pk.load(open('dataset.pkl','rb'))
data = data_set['training_data']
label= data_set['training_label']


kf = KFold(n_splits=10)
neighbors = list(range(1,20,2))
fold_accuracy= []
knn = KNeighborsClassifier()

for k in neighbors :
    if k == 1:
        for train_index, test_index in kf.split(data):
            data_train, data_test = data[train_index], data[test_index]
            label_train, label_test = label[train_index], label[test_index]
            # Fit the model on the training data
            knn.fit(data_train, label_train)
            # Evaluate the model on the testing data and calculate accuracy
            accuracy = knn.score(data_test, label_test)
            fold_accuracy.append(accuracy)
        
    knn.n_neighbors = k
    
    scores = cross_val_score(knn, data, label, cv=10)
    
    print('\nK:' ,k) 
    print('accuracy : %0.2f(+/-%0.2f)'%(scores.mean(),scores.std()*2))
    
print('\nK:',1)
for i, accuracy in enumerate(fold_accuracy):
    print(f"Accuracy for Fold {i+1}: {accuracy}")
