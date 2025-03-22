import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from models.cnn1.cnn import CNN
import torch.nn as nn
import torch.optim as optim

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
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    # Predict and output classification for first 10 images.
    for i in range(1,11):
        image_path = f"data/test/Image_{i}.jpg"

        model = CNN().to(device)
        model.load_state_dict(torch.load("cnn_model.pth"))

        predicted_class_name = classify_image(image_path, model, device)
        print(f"{image_path}: {predicted_class_name}")

if __name__ == "__main__":
    main()