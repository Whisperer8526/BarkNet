## About the project

These are my first steps in data science, machine learning and coding in general. Therefore, any kind of useful suggestions, remarks or opinions will be highly appreciated. Feel free to use the code and apply it to your own data. 

Everything is written in Python with use of classic machine learning libraries such as numpy, scikit-learn or tensorflow (see details in `Requirements.txt`).

The goal of this project is to explore possibilities and finally create small yet efficent model to classify tree species based on processed images of their bark. This approach was chosen due to the fact that other morphological features as e.g. leaves or buds are not always available throughout the year in temperate climare zone, or might be unreachable because of the tree height. Moreover, the process of gathering data in form of photos taken at the eye-level is much faster and makes collecting required dataset possible in reasonable amount of time. 

This choice, howerver, poses a challenge since bark of certain tree species may look almost identical and model is expected to lose some prediction accuracy especially between some pairs of species (e.g beech / hornbeam). On the other hand, species, such as birch are easily distinguishable even by untrained eye and shouldn't cause any major accuracy loss.

## Dataset 

Data has been collected in early spring of 2021, entirely in direct neighbourhood of Rzepin located in western part of Poland. These are species included in the project, they occur naturally in Central Europe:

  1. European beech (Fagus sylvatica)
  2. Silver birch (Betula pendula)
  3. Hornbeam (Carpinus betulus)
  4. Pedunculate oak (Quercus robur)
  5. Scots pine (Pinus sylverstris)

Images has been taken in production stands with moderately dense canopy. There is also a small variation of lighting conditions (time of day, overcast) within every subset belonging to single tree species. Every photo was taken at the height of 120-150 cm above the ground level. Most of them keep horizontal perspective but roughly 30% are pointing slightly up or down to add extra diversity. The bark takes no less than 70% of every image surface. 

![species](https://user-images.githubusercontent.com/75746226/117172744-60914980-adcc-11eb-932e-83e3f067c689.png)

Upon completion of data collection all images were renamed and resized to 224x224 pixels. This particular size was chosen, as it is commonly taken as input shape by many successful neural networks, which has been used as a point of reference in this project. Both image data and labels were converted into numpy arrays and saved as .npy files for further use. 

As mentioned before Hornbeam and Beech in created dataset proved to be most tricky to classify and below is an example of similarity in their bark texture.

![birch - hornbeam](https://user-images.githubusercontent.com/75746226/117205224-9f85c600-adf1-11eb-881a-edf4eaef808a.png)

## Data preprocessing

Two approaches were applied in terms of data preprocessing. For machine learning learning models such as SVC, numpy arrays with image data has been flattened to shape `(n_samples, 150528)` and pixel values normalized to range from 0 to 1. At the same time, since Convolutional Neural Network (CNN) requires 4D tensor input, the same data has been copied to shape `(n_samples, 224, 224, 3)`. After these steps whole dataset was divided to training set and test set with declared random state seed of 42 to make accuracy comparison possible betweeen models. Test set size was chosen to 20%.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    image_arrays, labels, test_size=0.2, random_state=42)
```

## Exploring models

Before finally choosing CNN couple as main model of machine learning algorythms were tested. They have been mostly trained with default parameters. Below you can find a list with the accuracy scores achieved by them. In some cases models were trained with black and white (single channel) images and are marked with as [BW]. Please check `Models Results.txt` for more detailed results. 

  1. SVC (kernel = rbf)                          0.67
  2. SVC (kernel = poly)                         0.65
  3. SVC (kernel = rbf, C=8, gamma=0.001) [BW]   0.63 
  4. Random Forest (n_estimators = 1000)         0.62
  5. AdaBoost [BW]                               0.47

#### Hyperparameters tuning

Due to limited computing power and time consuming nature of the task not much effort were spent on hyperparameters adjustment. Two kinds of cross validation search were applied to SVC (kernel= rbf) algorythm classifying black and white images. Initially a randomized search was conducted to obtain general estimate of 'C' and 'gamma' values: 

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'C': scipy.stats.expon(scale = 100), 
              'gamma': scipy.stats.expon(scale = .1),
              'kernel': ['rbf']}
random_search = RandomizedSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
random_search.fit(scaled_X_train, y_train)
```

and after getting these, more narrow grid search was used:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [8, 15, 30], 
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
grid.fit(scaled_X_train, y_train)
```

