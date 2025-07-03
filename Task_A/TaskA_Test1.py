import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your dataset

data_dir = "Task_A"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

train_dir = r"C:\Users\JCIN\OneDrive\Desktop\Comys_Hackathon5\Task_A\train"
val_dir = r"C:\Users\JCIN\OneDrive\Desktop\Comys_Hackathon5\Task_A\val"

# =====================
# Data Transforms
# =====================
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =====================
# Load Datasets
# =====================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =====================
# Handle Class Imbalance
# =====================
#class_counts = Counter([label for _, label in train_dataset])
#print(f"Class counts: {class_counts}")
#class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
#weights = torch.DoubleTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss()

# =====================
# Load Model (ResNet18)
# =====================
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================
# Training Function
# =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            # Print every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                current_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)
                current_acc = correct.double() / total
                print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {current_loss:.4f}, Accuracy: {current_acc:.4f}")

        # Epoch summary
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / len(train_loader.dataset)
        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate on validation set
        evaluate_model(model, val_loader)

# =====================
# Evaluation Function
# =====================
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision (male, female): {precision[0]:.4f}, {precision[1]:.4f}")
    print(f"Recall    (male, female): {recall[0]:.4f}, {recall[1]:.4f}")
    print(f"F1 Score  (male, female): {f1[0]:.4f}, {f1[1]:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=['male', 'female']))

# =====================
# Run Training
# =====================
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save model
torch.save(model.state_dict(), "gender_classifier.pth")
print("\nModel saved to 'gender_classifier.pth'")
