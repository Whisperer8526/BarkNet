import scipy                                                            #Randomized Search Cross-validation
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'C': scipy.stats.expon(scale=100), 
              'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf']}
random_search = RandomizedSearchCV(SVC(),param_grid,refit=True,verbose=2, cv=3)
random_search.fit(scaled_X_train, y_train)



from sklearn.svm import SVC                                             #Randomized Search Cross-validation
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf']}
grid = RandomizedSearchCV(SVC(),param_grid,refit=True,verbose=2, cv=3)
grid.fit(scaled_X_train, y_train)
