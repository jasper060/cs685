import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from models.cnn1.cnn import CNN
from utils.dataset import HumanActionDataset

# LOAD DATA 
def load_data(batch_size=64, train_csv='', test_csv='', train_dir='', test_dir=''):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
            )
    ])

    train_dataset = HumanActionDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    test_dataset  = HumanActionDataset(csv_file=test_csv, root_dir=test_dir, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# TRAIN MODEL
def train_model(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    print("training model")
    for epoch in range(epochs):
        running_loss = 0.0
        batch = 1
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"finished training batch {batch} in epoch {epoch}")
            batch+=1

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# TEST MODEL
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# MAIN 
def main():
    train_csv = 'data/Training_set.csv'
    test_csv  = 'data/Testing_set.csv'
    train_dir = 'data/train'
    test_dir  = 'data/test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    # Modify the classifier for 15 output classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 15)
    model = model.to(device)

    train_loader, test_loader = load_data(train_csv=train_csv, test_csv=test_csv, train_dir=train_dir, test_dir=test_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, device)
    

    # Save the trained model
    torch.save(model.state_dict(), "vgg16_model.pth")
    print("Model saved as vgg16_model.pth")

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()