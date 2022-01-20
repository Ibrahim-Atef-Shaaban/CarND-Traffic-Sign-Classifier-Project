# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Visualization.jpg "Visualization"
[image2]: ./Grayscale.jpg "Grayscaling"
[image4]: ./DownloadedGermanTraficSign/1.jpg "Traffic Sign 1"
[image5]: ./DownloadedGermanTraficSign/2.jpg "Traffic Sign 2"
[image6]: ./DownloadedGermanTraficSign/3.jpg "Traffic Sign 3"
[image7]: ./DownloadedGermanTraficSign/4.jpg "Traffic Sign 4"
[image8]: ./DownloadedGermanTraficSign/5.jpg "Traffic Sign 5"
[image9]: ./DownloadedGermanTraficSign/6.jpg "Traffic Sign 6"
[image10]: ./DownloadedGermanTraficSign/7.jpg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I displayed few images from the dataset with their corresponding labels on top of each image

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this helps greatly in reducing the processing time as instead of 3 depth layers, it only processes one. It was also mentioned in trafic signs classification articles that this is a better technique for classification.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because as mentioned in the lessons, this reduces the distribution of the onput data which makes it much more easier for the classifier to recognize what this sign is, without changing how the image look like.

I decided to generate additional data because ...  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flaten				| Input 5x5x16 Output 400  						|
| Fully connected		| Input 400 Output 170  						|
| Fully connected		| Input 170 Output 84   						|
| Dropout				| keepprob 0.65   								|
| Fully connected		| Input 84 Output 43   							|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a AdamOptimizer a batch size of 90, with number of epochs 50. I also set the learning rate to 0.0015, the mean of the weight random function to 0, standard diviation to 0.1.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of max = 99.784 % and final = 99.333 %
* validation set accuracy of max = 95.556 % and final = 94.490 % 
* test set accuracy of 91.7%

* The first architecture that was tried was the LeNet discribed in the lesons.
* Problems with the initial architecture, were that the validation accuracy was a bit below 90% which was not really good enough
* The architecture was adjusted by adding a dropout layer before the last fully connected layer calculation (the logits), ReLu activation function was used as it was refered to as the best activation function in the lecture from Stanford's CS231n course, also had to change one of the fully connected layers depth transformation from 120 to 170 for better results.
* I had to tune the Epochs, Batchsize and the learning rate, adjusted for a slightly higher learning rate, while reducing the sizes of the batch, in a larger number of Epochs, this way the model learn faster, without overfitting, or dropping far away from the best accuracy


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic Lights   		| Traffic Lights								| 
| work on road 			| work on road									|
| Children				| Children										|
| Ice warning     		| Ice warning					 				|
| Yield					| Yield		 									|
| Bumby Road			| Bumby Road 									|
| Merging right			| Merging right									|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.714 %. This compares favorably to the accuracy on the test set of 12630

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a traffic light sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Traffic Lights								| 
| 0.00    				| Speed limit (120km/h)							|
| 0.00					| General caution								|


For the second image, the model is sure that this is a work on the road sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| work on road  								| 
| 0.00    				| Turn left ahead 								|
| 0.00					| Keep right									|

For the third image, the model is sure that this is a Children crossing sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children										| 
| 0.00    				| Bicycles crossing 							|
| 0.00					| Slippery road									|

For the forth image, the model is relatively sure that this is a traffic light sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Slippery road									| 
| 0.02    				| Wild animals crossing							|
| 0.00					| Bicycles crossing								|



For the fifth image, the model is sure that this is a Yield sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield  										| 
| 0.00    				| Speed limit (20km/h)							|
| 0.00					| Speed limit (30km/h)							|

For the sixth image, the model is sure that this is a bumby road sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield  										| 
| 0.00    				| Bicycles crossing 							|
| 0.00					| Traffic signals								|


For the seventh image, the model is sure that this is a merge to the right sign (probability of 1.0).
The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield  										| 
| 0.00    				| Right-of-way at the next intersection			|
| 0.00					| Pedestrians									|
