import torch
import torchvision.transforms as transforms
from PIL import Image

def compute_mean_std(num_images):
    # Initialize variables to store cumulative sum of pixel values
    mean = torch.zeros(3)  # Assuming RGB images
    var = torch.zeros(3)
    
    # Define transformation to convert image to tensor
    to_tensor = transforms.ToTensor()

    # step I: Mean
    for idx in range(1, num_images+1):
        image = Image.open(f"data/test/Image_{idx}.jpg")
        image_tensor = to_tensor(image)
        mean += torch.mean(image_tensor, dim=(1, 2))

    mean /= num_images
    
    # step II: Std-dev
    # first we need mean from step I
    
    for idx in range(1, num_images+1):
        # Open image and convert to tensor
        image = Image.open(f"data/test/Image_{idx}.jpg")
        image_tensor = to_tensor(image)
        var += torch.mean((image_tensor - mean.unsqueeze(1).unsqueeze(2))**2, dim=(1, 2))
    
    return mean, torch.sqrt(var / num_images)

mean, std_dev = compute_mean_std(5400)

print(f"Mean of the data is: {mean}")
print(f"Standard deviation of the data is: {std_dev}")