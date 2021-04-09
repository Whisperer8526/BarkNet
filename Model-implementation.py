from sklearn.model_selection import train_test_split    #Splitting data

X_train, X_test, y_train, y_test = train_test_split(
    full_data['arrays'], full_data['labels'], test_size=0.2, random_state=42)


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
