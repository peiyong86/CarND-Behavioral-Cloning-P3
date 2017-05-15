#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/jitter.jpg "Jittered Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 / 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-31) 

The model includes RELU layers to introduce nonlinearity (model.py line 21-25), and the data is normalized in the model using a Keras lambda layer (model.py line 19). 

The model includes cropping layer to crop image (model.py line 20).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 27). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (src/readdata.py code line 75). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the model architecture from "End to End Learning for Self-Driving Cars" (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

I thought this model might be appropriate because it works in that paper and recomanded by udacity course.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I inserted a dropout layer to the model so that it will reduce overfitting problem.

Then I trained the model again, this time it does not overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I get more recovery from edge samples and added to training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-31) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 150x320x3 RGB image   							| 
| Lambda         		| lambda x: x / 255.0 - 0.5   							| 
| Cropping2D         		| cropping=((70,25), (0,0))   							| 
| Convolution 5x5     	| 2x2 stride, depth 24, activation="relu" |
| Convolution 5x5     	| 2x2 stride, depth 36, activation="relu" |
| Convolution 5x5     	| 2x2 stride, depth 48, activation="relu" |
| Convolution 3x3     	| 1x1 stride, depth 64, activation="relu" |
| Convolution 3x3     	| 1x1 stride, depth 64, activation="relu" |
| Flatten				|									|
| Dropout				| 0.5											|
| Fully connected		| outputs 100  					|
| Fully connected		| outputs 50  						|
| Fully connected		| outputs 10  						|
| Fully connected		| outputs 1  						|
|						|												|
 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also randomly shifted and rotated images thinking that this would enrich the data and reduce the overfitting. For example, here is an image that has then been augmented:

![alt text][image6]
![alt text][image7]

I didn't use flip because I collected data in both clockwise direction and counter clockwise direction, so my data already include both cases.

After the collection process, I had 46720 number of data points. 
I make a augmented data set by apply random shift (-5 to 5 pixels) and rotate (-10 to 10 degrees).
So the final number of data points are 93340.

I finally randomly shuffled the data set and put 33% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by "Behavioral Cloning Cheatsheet" from udacity course. I used an adam optimizer so that manually training the learning rate wasn't necessary.
