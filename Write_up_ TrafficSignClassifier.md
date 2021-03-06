**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Classifier**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Testimages/60.png 
[image5]: ./Testimages/ahead.png
[image6]: ./Testimages/giveway.png
[image7]: ./Testimages/noentry.png
[image8]: ./Testimages/nopassintruck.png
[image9]: ./Testimages/prioroad.png
[image10]: ./Testimages/stop.png
[image11]: ./data_dist.png
[image20]: ./features.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### The project code can be found [here](https://github.com/ChristerAkerblom/Udacity/blob/master/Traffic_Sign_Classifier-Grey%20scale-v2.ipynb) 
---

### Data Set Summary & Exploration
	
#### 1. Basic summary of the data set
I used the numpys library to calculate summary statistics of the traffic signs data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over different traffic signs. The distribution is similiar for both train, valid and test data sets.

![image 10](https://github.com/ChristerAkerblom/Udacity/blob/master/data_dist.png)

---
### Design and Test a Model Architecture

#### 1. Data preprocessing

As a first step, I decided to convert the images to grayscale since it showed slightly better performance on the validation data set and reduced number of weights. The data was the normalized to (-1,1) to improve backpropagation performance.  
I also decided to generate additional data because of two reasons:
* there were relative few samples for some signs
* reduce overfitting

I used scaling and rotation with a random number of the images since they are transformations that are not easily hanled by CNN architecture in contrast to e.g. translations. Here is an example of a traffic sign after grayscaling and normalizing and a copy being scaled and rotated randomly before the image was added to the data set.
![alt text](https://github.com/ChristerAkerblom/Udacity/blob/master/scale_rot.png)

For every image in the training set another augmented image was created hence doubling the size of the training set but keeping the distribution since it matched well with validation and test set.

#### 2. Model architecture 
Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        			| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         	| 32x32x1 gray scale image   				| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 		|
| RELU			|							|
| Max pooling	      	| 2x2 stride,  outputs 16x16x4, same padding 		|
| Dropout		| dropout probability 0.75				|
| Convolution 3x3	| 1x1 stride, same padding, outputs 10x10x16		|
| RELU			|							|
| Max pooling		| 2x2 stride, outputs 5x5x16, same padding 		|	
| Dropout		| dropout probability 0.75				|
| Flatten		| output 400 		     				|
| Fully connected	| output 120						|
| RELU			| 							|
| Fully connected	| output 84						|
| RELU			| 							|
| Fully connected	| output 43, no of classses				|
| RELU			| 							|
| Softmax		|         						|


#### 3. Training of the model
To train the model, I used an Adam optimizer with the following parameters.

* batch size: 128
* no of epochs: 50
* learning rate: 0.0005
* dropout probability: 0.75

#### 4. Approach taken to find feasable architecture 
My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.948
* test set accuracy of 0.934

To start the LeNet architecture from previous Udacity excersise was chosen. The architecture showed very good result on the training set but didn't generalize so well for the validation set indicating overfitting and dropout was introduced in order to reguralize the problem. Also reducing number of weights by working with grey scale images reduced overfitting. 

---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]
![alt text][image10]

The images are not real world examples and totally noise free, anyway intereting since this kind ofimages are not part of the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:-----------------------------:|:-------------------------------------:| 
| 60 km/h     			| 30 km/h  				| 
| Agead     			| Ahead 				|
| Yield				| Yield					|
| No entry	      		| No entry				|
| Truck no passing		| Truck no passing     			|
| Prio road			| Prio road				|
| Stop				| Stop					|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 86%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
All images except 60 km/h waspredicted with almot probability one and is not further disucussed. The top 5 probabilities for 60 km/h is presented below. 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .65         		| 30 km/h 	  				| 
| .30     		| Keep right					|
| .04			| 50 km/h					|
| .01	       		| 20 km/h					|
| 			| 120 km/h     					|
| 			| 60 km/h

For the second image ... 

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Visual output the network's feature maps
The trained network used edges to classify images. This can e.g. be seen in the images of the output from the first hidden layer with the priority road sign as input, see below.

![alt text][image20] 

