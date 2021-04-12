%matplotlib inline                                      #Displaying sample images from a dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

for i in range(1, 5):
    
    img_arr = X_train[random.choice(X_train.index)]
    img = img_arr.reshape(224, 224, 3)
    plt.subplot(2,2,i)
    plt.imshow(img)
    plt.axis("off")

full_data = create_dataset(directory, dst, label_dict, xls=False)    
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    full_data['arrays'], full_data['labels'], test_size=0.2, random_state=42)

import numpy as np

arr_X_train = np.vstack(X_train) #combining 
arr_X_test = np.vstack(X_test)
arr_y_train = np.vstack(y_train).ravel()
arr_y_test = np.vstack(y_test).ravel()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(arr_X_train)
scaled_X_test = scaler.fit_transform(arr_X_test)

from sklearn.svm import SVC
svc_clf = SVC(kernel='rbf', 
              class_weight='balanced', 
              verbose=True, 
              decision_function_shape='ovr')
              
svc_clf.fit(scaled_X_train, arr_y_train)

y_pred = svc_clf.predict(scaled_X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(arr_y_test,y_pred))

"""
[[ 1  0  9  0  1]
 [ 0 90  8  0  8]
 [ 0  3 73  0 16]
 [ 0  1  2  0  0]
 [ 0  4 29  0 67]]
              precision    recall  f1-score   support

           1       1.00      0.09      0.17        11
           2       0.92      0.85      0.88       106
           3       0.60      0.79      0.69        92
           4       0.00      0.00      0.00         3
           5       0.73      0.67      0.70       100

    accuracy                           0.74       312
   macro avg       0.65      0.48      0.49       312
weighted avg       0.76      0.74      0.73       312
"""
