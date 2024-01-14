import torch
import torch.nn as nn
from torchsummary import summary

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for conv1
        self.relu1 = nn.ReLU()  # ReLU activation after conv1

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for conv2
        self.relu2 = nn.ReLU()  # ReLU activation after conv2

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization for conv3
        self.relu3 = nn.ReLU()  # ReLU activation after conv3

        # Flatten layer to transition from convolutions to fully connected layers
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 64 channels, 28x28 image size -> 128 units
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout probability

        # Final output layer
        self.fc2 = nn.Linear(128, 10)  # 128 units to 10 classes (for MNIST digits)

    def forward(self, x):
        # Layer operations in forward pass
        x = self.relu1(self.bn1(self.conv1(x)))  # conv1 -> batch norm -> ReLU
        x = self.relu2(self.bn2(self.conv2(x)))  # conv2 -> batch norm -> ReLU
        x = self.relu3(self.bn3(self.conv3(x)))  # conv3 -> batch norm -> ReLU
        x = self.flatten(x)  # Flatten the output for fully connected layers
        x = self.dropout(self.relu1(self.fc1(x)))  # Fully connected -> ReLU -> Dropout
        x = self.fc2(x)  # Final output layer
        return x
# Create an instance of your model
model = ImageClassifier()

# Print the summary
summary(model, (1, 28, 28), device='cuda' if torch.cuda.is_available() else 'cpu')
