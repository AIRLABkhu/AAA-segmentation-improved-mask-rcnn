# Tittle
Automatic Detection and Segmentation of Thrombus in Abdominal Aortic Aneurysm Using Mask Region-based Convolutional Neural Network with Optimized Loss Functions

## Abstract
The detection and segmentation of thrombus regions is essential for monitoring disease progression on abdominal aortic aneurysm (AAA) and patient care and management. 
With the inherent capabilities to learn complex features, deep convolutional neural networks (CNNs) have been recently introduced to improve the thrombus detection and segmentation. Here, the investigation with CNN methods, however, is in an early stage, and most of the existing methods are heavily concerned with the segmentation of thrombus region, which only works after its detection process. 
In this work, we propose a fully automated method for the whole process of thrombus regions based on a well-established
mask region-based convolutional neural network (Mask R-CNN) architecture, where we improve the Mask R-CNN with optimized loss functions. 
Combined use of complete intersection over union (CIoU) and smooth L1 loss is designed for accurate thrombus detection, and the thrombus segmentation is improved with a modified focal loss. We evaluate our method using clinically approved 60 patient studies (i.e., computed tomography angiography (CTA) image volume data) by conducting 4-fold cross validation. Comparison results with multiple state-of-the-art methods suggest the superior performance of our method by achieving the highest F1 score of 0.9197 for the thrombus detection and outperforming most metrics in the thrombus segmentation.

## Result 

Overview of workflow for thrombus detection and segmentation using Mask R-CNN framework


![maskrcnn_re](https://user-images.githubusercontent.com/96813784/159687150-0388e330-b3d7-429f-af74-daeed6d18a34.png)



* * *
3D volume rendering results of the segmentation after using the detection algorithm. 
The overlapping part of ground truth and prediction mask from the model is represented in blue, False positive in blue, and False negative in red. Yellow box is represent the magnification region.


![화면 캡처 2022-03-23 200954](https://user-images.githubusercontent.com/96813784/159686837-f4c88d09-ff91-4545-97db-73c23d243d67.png)
