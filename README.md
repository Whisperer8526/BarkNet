# About the project

These are my first steps in data science, machine learning and coding in general. Therefore, any kind of useful suggestions, remarks or opinions will be highly appreciated. Feel free to use the code and apply it to your own data. 

Everything is written in Python with use of classic machine learning libraries such as numpy, scikit-learn or tensorflow (see details in "Requirements.txt").

The goal of this project is to explore possibilities and finally create small yet efficent model to classify tree species based on processed images of their bark. This approach was chosen due to the fact that other morphological features as e.g. leaves or buds are not always available throughout the year in temperate climare zone, or might be unreachable because of the tree height. Moreover, the process of gathering data in form of photos taken at the eye-level is much faster and makes collecting required dataset possible in reasonable amount of time. 

This choice, howerver, poses a challenge since bark of certain tree species may look almost identical and model is expected to lose some prediction accuracy especially between some pairs of species (e.g beech / hornbeam). On the other hand, species, such as birch are easily distinguishable even by untrained eye and shouldn't cause any major accuracy loss.

# Dataset 

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


