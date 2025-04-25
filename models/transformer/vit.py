import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import ViTFeatureExtractor, BitForImageClassification, TrainingArguments, Trainer
import evaluate
from PIL import Image
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor

from transformers import EarlyStoppingCallback


path_to_data = os.path.abspath('../../data')
train_dir = os.path.join(path_to_data, "train")
test_dir = os.path.join(path_to_data, "test")

data_dir = "temp"


# Load the model
# Hugging Face: Google / vit-base-patch16-224-in21K
model_id = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_id)

# Custom transformation pipeline for the dataset

def transform(image):
    inputs = image_processor(image, return_tensor="pt")
    return image["pixel_values"].squeeze(0) # remove batch for DataLoader

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

print(train_dataset)