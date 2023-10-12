## Landmark Classification and Tagging for Social Media

This project presents a simple weighted sum ensemble of pretrained CNN architectures to predict the most likely locations (out of 50 classes) where a landmark image was taken. The app will accept any user-supplied image as input and suggest the top 5 most relevant landmarks from 50 possible landmarks from across the world.


### Introduction

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

### Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
