
# Handwritten Digit Recognition with PyTorch and Pygame


## Overview
This repository contains a PyTorch-based model for handwritten digit recognition using the MNIST dataset. Additionally, there's a Pygame application that allows users to draw digits and obtain real-time predictions from the trained model.

## Project Structure
    model.py: Python script containing the PyTorch model architecture for digit recognition.
    train.py: Script for training the model on the MNIST dataset.
    predict.py: Script for making predictions using the trained model.
    app.py: Pygame application for drawing digits and obtaining predictions.
    requirements.txt: List of dependencies needed to run the project.

## Model Summary
```
-------------------------------------------------------------------------------------
Layer (type)        | Output Shape      | Param #     | Activation   | Additional Info
-------------------------------------------------------------------------------------
Conv2d-1            | [-1, 32, 28, 28]   | 320         | ReLU         | Kernel: (3x3)
BatchNorm2d-2       | [-1, 32, 28, 28]   | 64          |              | 
Conv2d-4            | [-1, 64, 28, 28]   | 18,496      | ReLU         | Kernel: (3x3)
BatchNorm2d-5       | [-1, 64, 28, 28]   | 128         |              | 
Conv2d-7            | [-1, 64, 28, 28]   | 36,928      | ReLU         | Kernel: (3x3)
BatchNorm2d-8       | [-1, 64, 28, 28]   | 128         |              | 
Flatten-10          | [-1, 50176]        | 0           |              | 
Linear-11           | [-1, 128]          | 6,422,656   | ReLU         | 
Dropout-13          | [-1, 128]          | 0           |              | Dropout: 50%
Linear-14           | [-1, 10]           | 1,290       |              | Output Layer
-------------------------------------------------------------------------------------
Total params: 6,480,010
Trainable params: 6,480,010
Non-trainable params: 0
-------------------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 3.26
Params size (MB): 24.72
Estimated Total Size (MB): 27.98
-------------------------------------------------------------------------------------
```

## Getting Started
```
#Clone the repository:
git clone https://github.com/your-username/your-repo.git
#Install dependencies:
pip install -r requirements.txt
#Train the model:
python train.py
#Run the Pygame application:
python app.py
```

# Usage
Use the drawing interface in the Pygame application to draw a digit.
Click "Predict" to obtain a real-time prediction from the trained model.