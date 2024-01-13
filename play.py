"""
    Name:       play.py
    Author:     Ayman Bouabra (abouabra)
    Created:    2023-01-13

    Purpose:    This program allows the user to draw a digit in the left box
                and the program will predict the digit and display the prediction in the right box.
                User can clear the drawing surface by clicking on the clear button.
"""

# import the necessary packages
import pygame
import numpy as np
import os
import cv2
import torch
from torch import nn

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

# Load the model
model_name = "MNIST_CUDA_MODEL.pt"
model = ImageClassifier()
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))


# Initialize Pygame
pygame.init()

# Set screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Digit Recognizer")

# Set background colors
background_color = (128, 128, 128)  # Gray
box_color = (0, 0, 0)  # White

# Create the drawing surface (box)
box = pygame.Surface((600, 600))
box.fill(box_color)
box_rect = box.get_rect(topleft=(0, 0))

# Create the clear button
button = pygame.Surface((200, 100))
button.fill((255, 0, 0))
button_rect = button.get_rect(topleft=(600, 0))

# Create the text on the button
button_font = pygame.font.SysFont(None, 50)
button_text = button_font.render("Clear", True, (0, 0, 0))
button_text_rect = button_text.get_rect(center=button_rect.center)

# Create the text for the prediction 
prediction_font = pygame.font.SysFont(None, 30)
prediction_text = []
prediction_text_rect = []
for i in range(10):
    prediction_text.append(prediction_font.render(f"{i}: 0%", True, (0, 0, 0)))
    prediction_text_rect.append(prediction_text[i].get_rect(center=(700, 150 + 30 * i)))


# Function to clear the drawing surface
def clear_box():
    box.fill(box_color)
    global prediction_text
    for i in range(10):
        prediction_text[i] = prediction_font.render(f"{i}: 0%", True, (0, 0, 0))

# Function to calculate softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def evaluate():
    try:
        # Load and preprocess image
        img = cv2.imread("box_image.png", cv2.IMREAD_GRAYSCALE)
        # Resize image to 28x28
        img = cv2.resize(img, (28, 28))
        # convert to tensor 
        img = torch.from_numpy(img)
        # Add batch dimension and normalize
        img = img.unsqueeze(0).unsqueeze(0).float() / 255.0  # Combine steps for clarity
        # Set model to evaluation mode
        model.eval()

        global prediction_text
        # Pass image through model
        with torch.no_grad():
            output = model(img)
            # Get prediction
            probabilities = softmax(output.numpy()[0])
            for i, probability in enumerate(probabilities):
                # round the probability to 2 decimal places
                accuracy = round(probability * 100, 2)
                prediction_text[i] = prediction_font.render(f"{i}: {accuracy}%", True, (0, 0, 0))
            # set the highest probability to green
            prediction_text[np.argmax(probabilities)] = prediction_font.render(f"{np.argmax(probabilities)}: {round(np.max(probabilities) * 100, 2)}%", True, (0, 255, 0))
        os.remove("box_image.png")  # Delete the image file
    except Exception as e:
        print("Error occurred:", e)  # Catch and print any errors

# Function to capture an image of the left box
def capture_box_image():
    # Create a new surface to capture the box's contents
    image_surface = pygame.Surface((box_rect.w, box_rect.h))
    image_surface.blit(box, (0, 0))  # Copy the box's pixels onto the image surface
    # save the image surface to a file
    pygame.image.save(image_surface, "box_image.png")



# Main loop
running = True
drawing = False
mouse_down_pos = None
while running:
    # Event handling
    for event in pygame.event.get():
        # Quit the program if the user closes the window
        if event.type == pygame.QUIT:
            running = False
        # Quit the program if the user presses escape
        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            running = False
        # Handle mouse events
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the user clicked on the drawing surface
            if box_rect.collidepoint(event.pos):
                drawing = True
                mouse_down_pos = event.pos
            # Check if the user clicked on the clear button
            elif button_rect.collidepoint(event.pos):
                clear_box()
        # Handle mouse motion while the user is clicking and dragging
        elif event.type == pygame.MOUSEMOTION and drawing:
            mouse_pos = event.pos
            pygame.draw.circle(box, (255, 255, 255), mouse_pos, 30)
            mouse_down_pos = mouse_pos
        # Handle mouse button release
        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            drawing = False
            capture_box_image()
            evaluate()
    
    # Draw everything
    screen.fill(background_color)
    screen.blit(box, box_rect)
    screen.blit(button, button_rect)
    screen.blit(button_text, button_text_rect)
    for i in range(10):
        screen.blit(prediction_text[i], prediction_text_rect[i])
    pygame.display.update()

# Quit the program
pygame.quit()
