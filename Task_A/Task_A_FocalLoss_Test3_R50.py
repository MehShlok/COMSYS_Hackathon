import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from datetime import datetime

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":

    # Paths
    train_dir = r"C:\Users\JCIN\OneDrive\Desktop\Comys_Hackathon5\Task_A\train"
    val_dir = r"C:\Users\JCIN\OneDrive\Desktop\Comys_Hackathon5\Task_A\val"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Compute class counts for weights
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset:
        class_counts[label] += 1
    class_weights = [1.0 / c for c in class_counts]
    alpha = torch.tensor(class_weights, dtype=torch.float32)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Setup
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Loss and Optimizer
    criterion = FocalLoss(alpha=alpha.to(device), gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Logging
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Training Loop
    num_epochs = 10
    best_val_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        cm_val = confusion_matrix(val_labels, val_preds)
        report_val = classification_report(val_labels, val_preds, target_names=val_dataset.classes, output_dict=True)

        log = []
        log.append(f"\n=== Epoch {epoch+1} ===")
        log.append(f"Train Loss           : {running_loss / len(train_loader.dataset)}")
        log.append(f"Val Accuracy         : {report_val['accuracy']}")
        log.append(f"Val Precision (female, male): {report_val['female']['precision']}, {report_val['male']['precision']}")
        log.append(f"Val Recall    (female, male): {report_val['female']['recall']}, {report_val['male']['recall']}")
        log.append(f"Val F1 Score  (female, male): {report_val['female']['f1-score']}, {report_val['male']['f1-score']}")
        log.append(f"Val Confusion Matrix:\n{cm_val}")

        print("\n".join(log))
        with open(log_file, 'a') as f:
            f.write("\n".join(log) + "\n")

        # Save best model
        avg_f1 = (report_val['female']['f1-score'] + report_val['male']['f1-score']) / 2
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), "best_gender_classifier_focal.pth")

    # Final Save
    torch.save(model.state_dict(), "final_gender_classifier_focal.pth")
