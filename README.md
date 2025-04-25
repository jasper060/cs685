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

### Data2

This is the second dataset. We might get rid of the first one, but for now we'll let it be. 

This training set also contains around 12,500 images, with the same classes. The difference was that the dataset came pre-divided in training and test sets. I've added a script to create a validation set as well. Each folder *(test, train, validation)* contains 15 sub-directories corresponding to an action/class. I've also included a script (*createLabels.py*) which traverses through each folder and writes the filenames present within into a csv file, along with the class label and the label number. The csv is in the same format as the **Training_set.csv** found in the first dataset

### Quickstart

## Set up virtual environment

I've used Python 3.13. You might have a different Python version. This needs to be finalized later, so that everyone is using the same Python version.

Run the following command to create a virtual environment: `python -m venv .venv`

This will create a virtual environment called **.venv**. You can keep any name you want. The (.) at the beginning of the directory name is to signify that it should be a hidden folder. If you use a different name for your virtual env, add it to the *.gitignore* file so that you do not accidentally upload it to the remote repository

If you're on Windows, activate the virtual env by typing in: `.venv/Scripts/activate`

If you're on Linux/MacOS, activate by typing: `.venv/bin/activate`

## Install required libraries

**Do this step after you've activated the virtual environment**

Type the following command to install libraries: `pip install -r requirements.txt`

If you use any new libraries, add to to the requirments.txt file by typing the following command: `pip freeze > requirements.txt`

## Fine-Tune the Resnet Model

1. Clone the repo
2. Run the **train_resnet.py** file. You can adjust the number of epochs in the file. This will take some time to run locally on CPU.
3. This will create a model on your local machine called **resnet_model.pth**
4. Run the **inference_resnet.py** file. It defaults to testing the first 10 images in the **test** dataset. Since this dataset is not currently labeled you can look at the output class and visually inspect if it's correct.

## Fine-Tune the ConvNext Model

1. Clone the repo
2. Run the **train_convnext.py** file. You can adjust the number of epochs in the file. This will take some time to run locally on CPU.
3. This will create a model on your local machine called **convnext_model.pth**
4. Run the **inference_convnext.py** file. It defaults to testing the first 36 images in the **test** dataset. Since this dataset is not currently labeled you can look at the output class and visually inspect if it's correct.
