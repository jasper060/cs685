import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import random

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

# classify an image with the given model path
def classify_image(num_images, model, device):
    image_list = []

    # Need to test on batch size of 36 to match training size
    for i in range(0, num_images):
      img_idx = i+1
      image_path = f"data/test/Image_{img_idx}.jpg"

      transform = transforms.Compose([
          transforms.Resize((128, 128)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      
      image = Image.open(image_path).convert('RGB')
      image = transform(image).unsqueeze(0).to(device)

      image_list.append(image)

    image_tensor = torch.cat(image_list, dim=0)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    print('predicted', predicted)
    return predicted

# MAIN 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.convnext_base(weights='DEFAULT').to(device)
    
    in_features = model.classifier[2].in_features
    num_classes = 15

    # Modify classifier
    model.classifier = nn.Sequential(
      nn.LayerNorm((36,1024,1,1)), 
      nn.Flatten(1),
      nn.Linear(in_features, num_classes)
    ).to(device)

    model.load_state_dict(torch.load("models/convnext/convnext_model.pth", map_location=torch.device("cpu")))

    predicted = classify_image(36, model, device)

    for idx, prediction in enumerate(predicted):
      print(f"Image {idx+1}: {class_names[prediction]}")

if __name__ == "__main__":
    main()