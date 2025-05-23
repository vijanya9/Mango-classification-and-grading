import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Define dataset paths
dataset_path = r"C:\Users\vijan\OneDrive\Desktop\mango_classifier\archive (2)\Dataset"
classification_dataset_path = os.path.join(dataset_path, "Classification_dataset")
grading_dataset_path = os.path.join(dataset_path, "Grading_dataset")

# Define transformations (similar to ImageDataGenerator)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (optional)
])

# Load Classification Dataset
classification_dataset = datasets.ImageFolder(root=classification_dataset_path, transform=transform)
classification_loader = DataLoader(classification_dataset, batch_size=32, shuffle=True)

# Load Grading Dataset
grading_dataset = datasets.ImageFolder(root=grading_dataset_path, transform=transform)
grading_loader = DataLoader(grading_dataset, batch_size=32, shuffle=True)

# Print class names
print("Classification Classes:", classification_dataset.class_to_idx)
print("Grading Classes:", grading_dataset.class_to_idx)

# Function to display images
import torchvision.utils as vutils

def show_images(loader, title, class_to_idx):
    # Get a batch of images
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Convert tensor images to numpy for plotting
    images = images.numpy().transpose((0, 2, 3, 1))  # Convert from (B, C, H, W) to (B, H, W, C)
    images = (images * 0.5) + 0.5  # Unnormalize (if normalization was applied)

    class_names = {v: k for k, v in class_to_idx.items()}  # Reverse class mapping

    plt.figure(figsize=(10, 5))

    for i in range(5):  # Show first 5 images
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(class_names[labels[i].item()])  # Convert tensor label to class name

    plt.suptitle(title)
    plt.show()

# Example usage
show_images(classification_loader, "Classification Dataset Samples", classification_dataset.class_to_idx)
show_images(grading_loader, "Grading Dataset Samples", grading_dataset.class_to_idx)