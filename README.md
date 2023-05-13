# Image-Segmentation-with-Clustering
The supervised approach (classification) classifies the data based on the model obtained from the training of labeled data. 
The supervised approach highly depends on the labeled training data and often require large number of training data set and 
high computational power during the model training phase. On the other hand, the unsupervised approach (clustering) groups the data without any the prior data training. 
This approach is relatively simpler to implement with lower computational cost. However, the accuracy of this technique highly depends on the initialization 
and the number of clusters chosen.

Examples: K-Means clustering, Fuzzy C-Means clustering, Mean shift clustering, Support Vector Machine (SVM), 

## Clustering VS Classification
| | Advantages | Disadvantages |
| --- | --- | --- |
| Clustering (Unsupervised) | - Doesn't require any prior knowledge of the data \newline - Suitable for evaluating new type of data- Can discover new data pattern- Simpler to implement


# Image Preprocessing
Image preprocessing has a significant impact on the clustering result especially when dealing with a highly detailed image and an image with complex structure. 
There are many image preprocessing techniques. Here, color space transforming, noise filtering, contrast stretching, image filling, and features adding have been performed. 

## Color space transforming
Color space transforming has a direct impact on the image structure. 
Different color spaces can emphasize different features in an image, making it easier to perform image segmentation based on colors. 
In this work, the HSV (Hue, Saturation, Value) color space has been selected as it help to better separate intensity values related to brightness of the image. 
In addition , the CIELAB (Luminance, a chrominance, b chrominance) color space has also been chosen to separate and group color together. 
It is not recommended to rely on one color space as this important features may be removed from some image. 
This issue can be fixed by features adding. The effect of transforming color space is shown in Figure 4. 
Color space transforming can be done in MATLAB using function **rgb2hsv(**) and **rgb2lab()** \\
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/7c5e91c4-6dad-4f0c-9a0c-0a419e7e7f98)|
|:--:|
|**Figure 4**: Image in different color space|

## Noise filtering
Noise in the image can be reduced by applying an image filter before the image processing. There are various filtering methods that can be applied to an image such as 
Averaging Filter, Gaussian lowpass filter, Circular Averaging Filter, and Prewitt filter.
After testing various filter, the Circular Averaging Filter (Disk Filter) has been chosen as it significantly reduces the image noise without removing the important features 
as shown in Figure 5. The Circular Averaging Filter can be done using MATLAB function **fspecial()** with **imfilter()** function \\
| ![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/6f711200-66cf-41d9-816c-7e43720900da) |
|:--:|
|**Figure 5**: Image with different noise filtering|

## Contrast stretching
Enhancing image’s contrast can significantly bring out the image features such as edges and color intensity. 
The goal of contrast stretching is to increase the dynamic range of an image so that it spans the full range of possible intensity values, from minimum to maximum. 
This technique works very well for image in the HSV color space as shown in **Figure 6**.  
Contrast stretching can be performed in MATLAB using function **imadjust()** \\
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/380fba02-83de-4601-98cd-8556391fbeb5)|
|:--:|
|**Figure 6**: The effect of contrast stretching on image in LAB and HSV color space|

## Image Filling
Image filling is a technique used to fill holes or missing regions in an image. This process is to increase the completeness of each segment of the image. 
The effect of image filling can be seen in **Figure 7**. Image filling can be done in MATLAB using function **imfill()** \\
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/1f546d54-7053-4b30-8fcc-5767700ad2b3)|
|:--:|
|**Figure 7**: The effect of image filling on image in LAB and HSV color space|

## Features adding
In this work, all the aforementioned preprocessing methods are combined before feeding into the clustering algorithm as shown in **Figure 8**. 
An image has been transformed into HSV and LAB color space and undergoes noise filtering, contrast stretching and image filling. 
However, features adding should be used carefully, as adding more features does not necessarily give better result. \\
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/c9609f01-6c4e-431d-ac94-e7b90a432a6e)|
|:--:|
|**Figure 8**: Overall image preprocessing|

# Evaluation Algorithm
As the ground truth and the segmented image may have different number of clusters, it is necessary to create a cluster matching algorithm 
before evaluating the segmenting performance. The overall algorithm is as follow:

![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/ecadb916-612a-4518-9761-cdfeb3548c09)

In this work, the Jaccard similarity has been used and the similarity threshold has been set to be
≥40%. This algorithm has been custom coded in a function **binaryMatch()** in MATLAB.





# References
[1] T. Verma and N. Patel, "Data Clustering: Algorithms and Applications", Springer, 2020. 

[2] S. Kumar. “C-Means Clustering Explained” Builtin.  https://builtin.com/data-science/c-means (accessed Feb. 5, 2023)

[3] S. Ghosh and  S. Kumar Dubey “Comparative Analysis of K-Means and Fuzzy C-Means Algorithms”, IJACSA, 2013 







