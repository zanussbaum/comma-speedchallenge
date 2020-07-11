# comma-speedchallenge
This is my attempt at the comma.ai Speed Challenge.

## Background
As illustrated by [Ryan](https://github.com/ryanchesler/comma-speed-challenge), there are many different approaches to this problem. Optical flow and LSTMs seem to be pervasive in pretty much everyone's approach, if they chose to use a deep learning approach. Many of the models seem to overfit, due to lack of training data (there's only 20400) frames in total. So, we will attempt to use transfer learning to achieve good results.