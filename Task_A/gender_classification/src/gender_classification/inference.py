import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

def load_model(weights_path, num_classes=2, device='cpu'):
    if not os.path.isfile(weights_path):
        print(f"Error: weights file not found at '{weights_path}'")
        sys.exit(1)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

def main():
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights (.pth)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with explicit weights path
    model = load_model(args.weights_path, device=device)

    # Run prediction
    prediction = predict_image(model, args.image_path, device=device)

    # Assuming your training classes were ['female', 'male']
    classes = ['female', 'male']
    print(f"Predicted Class: {classes[prediction]}")

if __name__ == "__main__":
    main()
