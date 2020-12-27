# Cat vs Dog Image Classifier
Compares the accuracy of KNN, HOG/SVM and CNN for classifying an image as cat or dog.  

# Conclusion  
A CNN is the best approach to this dataset with an 89.5% accuracy.  
Neither the KNN or HOG/SVM performed well enough to be considered useable for this dataset as they barely did better than a random guess.  

# Analysis of Each Algorithm (best to worst)
## CNN (Convolutional Neural Network)
CNN written using Pytorch.   
### CNN Results
A model with an accuracy of 89.5% was created using 11 convolutional layers, relu activation functions, batch normalzation between each convoltuion and one max pooling layer.  
### CNN Setup Instructions
[put the train data set from this link - https://www.kaggle.com/c/dogs-vs-cats/data - ](https://www.kaggle.com/c/dogs-vs-cats/data) into the following directories:  
dataYouTubeFormat/train/cat  
dataYouTubeFormat/train/dog  
dataYouTubeFormat/test/cat  
dataYouTubeFormat/test/dog  
Some hyperparameters can be set under `Constants` in CNN_CatVsDog.py, while others will need to be set in the `ConvNN` class or in the following lines of code:  
```
transformer = transforms.Compose([
    transforms.Resize( (150,150) ), # Is this the correct size?
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## HOG / SVM (Histogram of Oriented Gradients / Linear SVM)  
HOG / SVM written using scikit-learn.  
### HOG / SVM Results
The accuracy of the HOG / SVM algorithm consistently got around a 60% accuracy even with many different hyperparameters and training set sizes.  
### KNN Setup Instructions
[put the train data set from this link - https://www.kaggle.com/c/dogs-vs-cats/data - into the directory data/](https://www.kaggle.com/c/dogs-vs-cats/data)  
Hyperparameters for the SVM can be set under `Constants` in HOG-CatDog.py and the following 2 lines for the bin size:  
```
fd = fd.round(1)
...
for i in np.arange(0, 1.1, 0.1).round(1).tolist():
```

Hyperparameters for the HOG need to be set in the following line of code within HOG-CatDog.py:  
`fd, hogImage = hog(image, orientations=64, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True, multichannel=True)`

## KNN (k-nearest neighbors)
KNN written from scratch using Python3.  
### KNN Results
The accuracy was around 50%-60%.  
KNN was tested for k = 3, 7, 11, 23, 45, 101, 201 and 301.  
### KNN Setup Instructions
[put the train data set from this link - https://www.kaggle.com/c/dogs-vs-cats/data - into the directory data/](https://www.kaggle.com/c/dogs-vs-cats/data)  
Hyperparameters can be set under `Constants` in knn_catVsDog.py  
