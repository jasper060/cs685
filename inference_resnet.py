import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from models.cnn1.cnn import CNN
import torch.nn as nn
import torch.optim as optim
import random

# classify an image with the given model path
def classify_image(image_path, model, device):
    class_names = [
        'calling',
        'clapping',
        'cycling',
        'dancing',
        'drinking',
        'eating',
        'fightning',
        'hugging',
        'laughing',
        'listening_to_music',
        'running',
        'sitting',
        'sleeping',
        'texting',
        'using_laptop'
    ]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5729, 0.5379, 0.5069), (0.3056, 0.3022, 0.3096))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# MAIN 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predict and output classification for 10 randomly selected images.
    for i in range(1,11):
        img_idx = random.randint(1, 5400)
        image_path = f"data/test/Image_{img_idx}.jpg"

        model = models.resnet152()
        model.fc = nn.Linear(model.fc.in_features, 15)
        model.load_state_dict(torch.load("models/resnet/resnet_model.pth",map_location=torch.device("cpu")))
        model.to(device)

        predicted_class_name = classify_image(image_path, model, device)
        print(f"{image_path}: {predicted_class_name}")

if __name__ == "__main__":
    main()