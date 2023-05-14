# Image-Segmentation-with-Clustering
# Introduction
The supervised approach (classification) classifies the data based on the model obtained from the training of labeled data. 
The supervised approach highly depends on the labeled training data and often require large number of training data set and 
high computational power during the model training phase. On the other hand, the unsupervised approach (clustering) groups the data without any the prior data training. 
This approach is relatively simpler to implement with lower computational cost. However, the accuracy of this technique highly depends on the initialization 
and the number of clusters chosen.

Examples: K-Means clustering, Fuzzy C-Means clustering, Mean shift clustering, Support Vector Machine (SVM), 

## Clustering VS Classification
![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/8c174b56-c4ad-45f2-8c60-9c1426c2e728)

## Comparison between different clustering techniques
![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/651e8aaa-74a5-4dba-bbe0-bed782b1e737)



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
Color space transforming can be done in MATLAB using function **rgb2hsv(**) and **rgb2lab()** 

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/7c5e91c4-6dad-4f0c-9a0c-0a419e7e7f98)|
|:--:|
|**Figure 4**: Image in different color space|

## Noise filtering
Noise in the image can be reduced by applying an image filter before the image processing. There are various filtering methods that can be applied to an image such as 
Averaging Filter, Gaussian lowpass filter, Circular Averaging Filter, and Prewitt filter.
After testing various filter, the Circular Averaging Filter (Disk Filter) has been chosen as it significantly reduces the image noise without removing the important features 
as shown in Figure 5. The Circular Averaging Filter can be done using MATLAB function **fspecial()** with **imfilter()** function

| ![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/6f711200-66cf-41d9-816c-7e43720900da) |
|:--:|
|**Figure 5**: Image with different noise filtering|

## Contrast stretching
Enhancing image’s contrast can significantly bring out the image features such as edges and color intensity. 
The goal of contrast stretching is to increase the dynamic range of an image so that it spans the full range of possible intensity values, from minimum to maximum. 
This technique works very well for image in the HSV color space as shown in **Figure 6**.  
Contrast stretching can be performed in MATLAB using function **imadjust()**

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/380fba02-83de-4601-98cd-8556391fbeb5)|
|:--:|
|**Figure 6**: The effect of contrast stretching on image in LAB and HSV color space|

## Image Filling
Image filling is a technique used to fill holes or missing regions in an image. This process is to increase the completeness of each segment of the image. 
The effect of image filling can be seen in **Figure 7**. Image filling can be done in MATLAB using function **imfill()** 

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/1f546d54-7053-4b30-8fcc-5767700ad2b3)|
|:--:|
|**Figure 7**: The effect of image filling on image in LAB and HSV color space|

## Features adding
In this work, all the aforementioned preprocessing methods are combined before feeding into the clustering algorithm as shown in **Figure 8**. 
An image has been transformed into HSV and LAB color space and undergoes noise filtering, contrast stretching and image filling. 
However, features adding should be used carefully, as adding more features does not necessarily give better result. 

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/c9609f01-6c4e-431d-ac94-e7b90a432a6e)|
|:--:|
|**Figure 8**: Overall image preprocessing|

# Evaluation Algorithm
As the ground truth and the segmented image may have different number of clusters, it is necessary to create a cluster matching algorithm 
before evaluating the segmenting performance. The overall algorithm is as follow:

![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/ecadb916-612a-4518-9761-cdfeb3548c09)

In this work, the Jaccard similarity has been used and the similarity threshold has been set to be
≥40%. This algorithm has been custom coded in a function **binaryMatch()** in MATLAB.

https://github.com/komxun/Image-Segmentation-with-Clustering/blob/43d9bd44285e1812698b5dd94de3026bbaa2c357/Main_Clustering.m#L379-L466

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/61d84a89-97fe-4bd9-9e25-c6780178044b)|
|:--:|
|**Figure 10**: Clusters matching algorithm scheme|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/668a1ae4-f0e0-4882-996f-0717310368ab)|
|:--:|
|**Figure 11**: Segmentation evaluation scheme for each image|

# Results
## Segmentation Results
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/21f27b7a-267f-4d04-a412-8888db990afe)|
|:--:|
|**Figure 14**: Segmentation results for image 1|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/24fd156f-b19d-4bb7-96a8-06541894a20a)|
|:--:|
|**Figure 15**: Segmentation results for image 2|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/c6d36dcd-4de6-4e08-b963-33ed17b4c2a1)|
|:--:|
|**Figure 16**: Segmentation results for image 3|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/0f748108-d528-47e9-ba00-23f3e1d567ae)|
|:--:|
|**Figure 17**: Segmentation results for image 4|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/b34d5b44-a7aa-4cde-8c89-3b8d2a96bda0)|
|:--:|
|**Figure 18**: Segmentation results for image 5|

## Evaluation Results
Here, only the cluster matching of the FCM technique has been plotted since the clusters from Mean Shift clustering are highly detailed and the number of clusters also vary drastically. 
|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/2eae3f71-3fae-4a33-bdf0-79d1ed4d569a)|
|:--:|
|**Figure 19**: Cluster matching result for image 1 with FCM-based segmentation|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/d21a1878-6777-430f-a125-8f555dd093c0)|
|:--:|
|**Figure 20**: Cluster matching result for image 2 with FCM-based segmentation|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/aee7c39c-7d32-4844-a5e9-e0f163a02f7f)|
|:--:|
|**Figure 21**: Cluster matching result for image 3 with FCM-based segmentation|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/fd900b44-e7b9-4ff2-9fb5-85a39d36528d)|
|:--:|
|**Figure 22**: Cluster matching result for image 4 with FCM-based segmentation|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/fad77b73-6077-4c0b-863c-57744c57bb21)|
|:--:|
|**Figure 23**: Cluster matching result for image 5 with FCM-based segmentation|

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/a2768b88-a9d5-4866-9c3f-04ace399c0e3)|
|:--:|
|**Figure 24**: Segmentation performance scores for FCM and Mean shift clustering technique on five images|

# Evaluation Results Discussion
It can be seen from Figure 19 - Figure 23 that the clusters matching algorithm works very well in matching the obtained segments with ground truth segments, 
making the evaluated scores highly reliable. According to Figure 24,by evaluating both techniques for 5 images, FCM clustering technique gave higher performance as expected. 

Nonetheless, as shown Figure 25, after evaluating for 200 images, the Mean Shift clustering techniques gave higher precision and accuracy than FCM. This implies the result from Mean Shift clustering has low number of False Positive (FP) data. This is because the number of clusters for the FCM clustering has been fixed to three clusters which is a very strict constraints, while the number of cluster in Mean Shift clustering is determined automatically. In spite of that, FCM clustering still has the highest average recall, indicating that FCM has higher number of True Positive (TP) data and lower False Negative (FN) data.

|![image](https://github.com/komxun/Image-Segmentation-with-Clustering/assets/133139057/5d218830-1643-4eaf-8cfb-b16f2dd10db3)|
|:--:|
|**Figure 25**: Segmentation performance evaluated for 200 images|

# Conclusion
K-means clustering technique should only be used if the computational cost is at concern. Fuzzy C-Means (FCM) clustering gives the most consistent performance if a prior information of the image is known, or if the number of segmentations can be approximated. Lastly, Mean Shift clustering technique has the highest flexibility and is most suitable for segmenting an unknown image.

Image preprocessing can significantly improve the performance for any segmentation techniques. However, it should be done carefully as some process may remove important image features resulting in a wrong segmentation.





# References
[1] T. Verma and N. Patel, "Data Clustering: Algorithms and Applications", Springer, 2020. 

[2] S. Kumar. “C-Means Clustering Explained” Builtin.  https://builtin.com/data-science/c-means (accessed Feb. 5, 2023)

[3] S. Ghosh and  S. Kumar Dubey “Comparative Analysis of K-Means and Fuzzy C-Means Algorithms”, IJACSA, 2013 







