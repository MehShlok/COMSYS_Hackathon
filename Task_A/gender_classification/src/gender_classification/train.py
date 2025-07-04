import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from .focal_loss import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train gender classification model with Focal Loss")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n✅ Checking data directories...")
    if not os.path.isdir(args.train_dir):
        raise ValueError(f"Training directory not found: {args.train_dir}")
    if not os.path.isdir(args.val_dir):
        raise ValueError(f"Validation directory not found: {args.val_dir}")
    print(f" - Training data: {args.train_dir}")
    print(f" - Validation data: {args.val_dir}")

    print("\n✅ Preparing data transformations and loading datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform)
    print(f" - Found {len(train_dataset)} training images in {len(train_dataset.classes)} classes")
    print(f" - Found {len(val_dataset)} validation images in {len(val_dataset.classes)} classes")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("\n✅ Computing class weights for Focal Loss...")
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset:
        class_counts[label] += 1
    class_weights = [1.0 / c for c in class_counts]
    alpha = torch.tensor(class_weights, dtype=torch.float32)
    print(f" - Class counts: {class_counts}")
    print(f" - Class weights (alpha): {alpha.tolist()}")

    print("\n✅ Setting up model, criterion, optimizer and scheduler...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - Using device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    print(" - Model loaded and customized")

    criterion = FocalLoss(alpha=alpha.to(device), gamma=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    print(" - Loss, optimizer, scheduler initialized")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    log_file = os.path.join("logs", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    print(f" - Logs will be saved to: {log_file}")

    num_epochs = 10
    best_val_f1 = 0

    print("\n🚀 Starting training...\n")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []

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
        print(f" - Scheduler step completed. Current LR: {scheduler.get_last_lr()}")

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
        log.append(f"Train Loss: {running_loss / len(train_loader.dataset):.4f}")
        log.append(f"Val Acc   : {report_val['accuracy']:.4f}")
        log.append(f"Val F1    : {report_val['female']['f1-score']:.4f}, {report_val['male']['f1-score']:.4f}")
        log.append(f"Val CM    :\n{cm_val}")

        print("\n".join(log))
        with open(log_file, 'a') as f:
            f.write("\n".join(log) + "\n")

        avg_f1 = (report_val['female']['f1-score'] + report_val['male']['f1-score']) / 2
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), "saved_models/best_gender_classifier_focal.pth")
            print("✅ Saved new best model weights.")

    torch.save(model.state_dict(), "saved_models/final_gender_classifier_focal.pth")
    print("\n🎉 Training complete. Final model saved.")

if __name__ == "__main__":
    main()
