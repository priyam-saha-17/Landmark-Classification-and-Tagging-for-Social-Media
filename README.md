## Landmark Classification and Tagging for Social Media

This project presents a simple weighted sum ensemble of pretrained CNN architectures to predict the most likely locations (out of 50 classes) where a landmark image was taken. The app will accept any user-supplied image as input and suggest the top 5 most relevant landmarks from 50 possible landmarks from across the world.


### Introduction

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

### Method

The cnn_from_scratch.ipynb implements a CNN architecture from scratch that achieves a classification accuracy of 59%.

The transfer_learning.ipynb implements 4 pretrained CNN architectures --- ResNet18, VGG16, EfficientNetb0 and MobileNetV2, which achieves a test accuracy of 73%, 68%, 72% and 72% respectively. Thereafter, a weighted sum ensemble is implemented using these 4 base learners that achieved a test accuracy of 81.6% . The optimised weights corresponding to each base learner was obtained by running a grid search algorithm.

The app.ipynb implements an application which shows the top 5 classes that the ensemble model thinks are the most relevant for the picture a user has uploaded on the application.

The src folder contains all the necessary python files for data loading, CNN architecture from scratch, the transfer models etc.


### Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.

### Credits

This project is a part of the Udacity ML Fundamentals Nanodegree program, which I received as part of the AWS AI-ML Scholarship 2022-23.
