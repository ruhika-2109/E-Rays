import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.images_dir = self.root_dir / split / 'images'
        self.labels_dir = self.root_dir / split / 'labels'
        
        print(f"Looking for images in: {self.images_dir}")
        print(f"Path exists: {self.images_dir.exists()}")
        
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.png')) +
                                 list(self.images_dir.glob('*.jpeg')) +
                                 list(self.images_dir.glob('*.JPG')) +
                                 list(self.images_dir.glob('*.PNG')) +
                                 list(self.images_dir.glob('*.JPEG')))
        
        print(f"{split} set: {len(self.image_files)} images")
        if len(self.image_files) == 0 and self.images_dir.exists():
            print(f"Warning: No images found. Files in directory:")
            print(list(self.images_dir.iterdir())[:5])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        label_path = self.labels_dir / (img_path.stem + '.txt')
        has_fracture = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                has_fracture = 1 if len(lines) > 0 else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, has_fracture

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class BoneFractureCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BoneFractureCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def create_resnet_model(num_classes=2, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(data_root, model_type='custom', num_epochs=50, batch_size=16, lr=0.001):
    train_dataset = BoneFractureDataset(data_root, 'train', train_transform)
    val_dataset = BoneFractureDataset(data_root, 'valid', val_test_transform)
    test_dataset = BoneFractureDataset(data_root, 'test', val_test_transform)
    
    empty_splits = []
    if len(train_dataset) == 0:
        empty_splits.append(str(Path(data_root) / 'train' / 'images'))
    if len(val_dataset) == 0:
        empty_splits.append(str(Path(data_root) / 'valid' / 'images'))
    if len(test_dataset) == 0:
        empty_splits.append(str(Path(data_root) / 'test' / 'images'))
    if empty_splits:
        raise RuntimeError(
            "No images found in expected dataset folders.\n"
            f"Empty or missing: {empty_splits}\n"
            "Expected structure: <data_root>/train/images, <data_root>/valid/images, <data_root>/test/images"
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if model_type == 'resnet':
        model = create_resnet_model(num_classes=2, pretrained=True)
        print("Using ResNet50 with transfer learning")
    else:
        model = BoneFractureCNN(num_classes=2)
        print("Using custom CNN architecture")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = 'best_bone_fracture_model.pth'
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")
    
    model.load_state_dict(torch.load(best_model_path))
    
    print("\n" + "="*60)
    print("Testing on test set...")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['No Fracture', 'Fracture']))
    
    plot_history(history)
    plot_confusion_matrix(test_labels, test_preds)
    
    return model, history

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fracture', 'Fracture'],
                yticklabels=['No Fracture', 'Fracture'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train bone fracture model")
    parser.add_argument('--data_root', type=str, default=None, help='Path to dataset root folder')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        candidates = [
            script_dir / "Bone Fractures Detection",
            Path.cwd() / "Bone Fractures Detection",
            script_dir.parent / "Bone Fractures Detection",
            script_dir / "Human Bone Fractures Multi-modal Image Dataset",
            Path(r"D:\Technical Projects\SWHACK ganavi\fractures") / "Bone Fractures Detection",
        ]
        data_root = None
        for c in candidates:
            if c.exists():
                data_root = c
                break
        if data_root is None:
            print("ERROR: Could not find dataset in common locations. Provide --data_root or place folder in one of:")
            for c in candidates:
                print("  ", c)
            print("\nScript folder listing:")
            for p in sorted(script_dir.iterdir()):
                print("  ", p.name)
            raise SystemExit(1)

    data_root = str(data_root)
     
    print(f"Dataset root: {data_root}")
    print(f"Current working directory: {os.getcwd()}")
     
    model, history = train_model(
        data_root=data_root,
        model_type='resnet',
        num_epochs=50,
        batch_size=16,
        lr=0.0001
    )
     
    print("\nTraining completed!")
    print("Model saved as: best_bone_fracture_model.pth")