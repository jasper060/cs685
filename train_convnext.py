import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader
from PIL import Image
from utils.dataset import HumanActionDataset

# LOAD DATA 
def load_data(batch_size=36, train_csv='', test_csv='', train_dir='', test_dir=''):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = HumanActionDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    test_dataset  = HumanActionDataset(csv_file=test_csv, root_dir=test_dir, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# TRAIN MODEL
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    print("training model")
    for epoch in range(epochs):
        print(f"in epoch {epoch}")
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
    
    model = models.convnext_base(weights='DEFAULT').to(device)

    for param in model.parameters():
        param.requires_grad = False
        
    # Get number of features in classifier
    in_features = model.classifier[2].in_features
    
    # Save original number of classes
    num_classes = 15 
    
    # Modify classifier for fine tuning.
    model.classifier = nn.Sequential(
        nn.LayerNorm((36,1024,1,1)),  # Keep the first layer
        nn.Flatten(1),
        nn.Linear(in_features, num_classes)  # Replace with your class count
    ).to(device)

    train_loader, test_loader = load_data(train_csv=train_csv, test_csv=test_csv, train_dir=train_dir, test_dir=test_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, device)
    #test_model(model, test_loader, device)

    # Save the trained model
    torch.save(model.state_dict(), "convnext_model.pth")
    print("Model saved as convnext_model.pth")

if __name__ == "__main__":
    main()