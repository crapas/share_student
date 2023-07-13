import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

lib_path = os.environ.get('LIBPATH')
if lib_path is None:
    lib_path = '../lib/'


import sys
sys.path.append(lib_path)
from digit_recognizer import DigitRecognizer


# WHAT TO DO : 1. 학습과 관련없는 것은 다 지웠음. 2. MODEL/DATA PATH 설정




model_path = os.environ.get('MODELPATH')
if model_path is None:
    model_path = '../model/'

data_path = os.environ.get('DATAPATH')
if data_path is None:
    data_path = "../data/"
    
# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define transforms
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

# Load the dataset
#data_dir = "data"  # Directory containing the digit images
dataset = ImageFolder(data_path, transform=transform)

# Calculate train-test split
total_size = len(dataset)
train_size = int(0.9 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = DigitRecognizer().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")



torch.save(model.state_dict(), model_path + 'modelfile')
