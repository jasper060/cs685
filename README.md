# cs685

CS 685 Data Science Project

## Overview

The intention of this project is to design, analyze, and compare some image classification ML models.

The specific types of images
we are classifying are **Human Action** images that fall into one of 15 classes. Each image contains one class label. The classes
are:

- calling
- clapping
- cycling
- dancing
- drinking
- eating
- fightning
- hugging
- laughing
- listening_to_music
- running
- sitting
- sleeping
- texting
- using_laptop

## Getting Started

### Data

There is a training dataset and a test dataset. The training dataset contain 12,600 images and the test dataset contains 5,400 images.

There is a file named **training_dataset.csv** which contains the filename and the label of each image in the training dataset.

The images in the test dataset do not have labels.

### Quickstart

## Fine-Tune the Resnet Model

1. Clone the repo
2. Run the **train_resnet.py** file. You can adjust the number of epochs in the file. This will take some time to run locally on CPU.
3. This will create a model on your local machine called **resnet_model.pth**
4. Run the **inference_resnet.py** file. It defaults to testing the first 10 images in the **test** dataset. Since this dataset is not currently labeled you can look at the output class and visually inspect if it's correct.

5. ## Fine-Tune the ConvNext Model

1. Clone the repo
2. Run the **train_convnext.py** file. You can adjust the number of epochs in the file. This will take some time to run locally on CPU.
3. This will create a model on your local machine called **resnet_model.pth**
4. Run the **inference_convnext.py** file. It defaults to testing the first 10 images in the **test** dataset. Since this dataset is not currently labeled you can look at the output class and visually inspect if it's correct.
