import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# --- GPU Detection ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    print("⚠️ No GPU detected, using CPU")

# --- Dataset Path ---
path = r'd:\Technical Projects\SWHACK ganavi\tumor'

# --- Build DataFrames ---
def build_df(base_path):
    data = []
    for label in os.listdir(base_path):
        label_dir = os.path.join(base_path, label)
        if os.path.isdir(label_dir):
            for img in os.listdir(label_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'Class Path': os.path.join(label_dir, img),
                        'Class': label
                    })
    return pd.DataFrame(data)

print("📁 Building dataframes...")
train_full_df = build_df(os.path.join(path, 'Training'))
test_df = build_df(os.path.join(path, 'Testing'))

# Check class distribution
print("\n📊 Training Class Distribution:")
print(train_full_df['Class'].value_counts())
print(f"\nTotal Training samples: {len(train_full_df)}")
print(f"Total Testing samples: {len(test_df)}")

train_df, valid_df = train_test_split(
    train_full_df, test_size=0.2, random_state=42, stratify=train_full_df['Class']
)

print(f"\nTrain/Validation split: {len(train_df)}/{len(valid_df)}")

# --- Dataset Class ---
class TumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(dataframe['Class'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Classes: {self.classes}")
        print(f"Class mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'Class Path']
        try:
            img = Image.open(img_path).convert('RGB')
            label = self.class_to_idx[self.dataframe.loc[idx, 'Class']]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return next valid image instead of dummy
            return self.__getitem__((idx + 1) % len(self))

# --- Enhanced Data Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Dataloaders ---
batch_size = 16  # Smaller batch for better gradient updates
print("\n🔄 Creating data loaders...")

train_dataset = TumorDataset(train_df, train_transform)
valid_dataset = TumorDataset(valid_df, test_transform)
test_dataset = TumorDataset(test_df, test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=0, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                         num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=0, pin_memory=True)

classes = train_dataset.classes
num_classes = len(classes)
print(f"\n✅ Number of classes: {num_classes}")

# --- Improved Model Definition ---
class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use EfficientNet-B3
        self.base_model = models.efficientnet_b3(weights='IMAGENET1K_V1')

        # Unfreeze more layers for better learning
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last 40 parameters instead of 20
        for param in list(self.base_model.parameters())[-40:]:
            param.requires_grad = True

        num_features = self.base_model.classifier[1].in_features
        
        # Improved classifier head
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

print("🔄 Initializing model...")
model = TumorClassifier(num_classes=num_classes).to(device)
print(f"✅ Model initialized on {device}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")

# --- Calculate Class Weights for Imbalanced Data ---
class_counts = Counter(train_df['Class'])
total_samples = len(train_df)
class_weights = torch.tensor([
    total_samples / (num_classes * class_counts[cls]) 
    for cls in classes
], dtype=torch.float32).to(device)

print(f"\n⚖️ Class weights: {dict(zip(classes, class_weights.cpu().numpy()))}")

# --- Training Setup ---
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True
)

# --- Train & Validation Functions ---
def train_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}, Acc: {correct/total*100:.2f}%")
    
    return total_loss / total, correct / total

def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels

# --- Training Loop ---
num_epochs = 30
best_val_acc = 0
patience = 7
patience_counter = 0

print("\n" + "="*60)
print("🚀 Starting Training...")
print("="*60 + "\n")

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print('='*60)
    
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc, val_preds, val_labels = validate(model, valid_loader)
    scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n📊 Results:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print(f"  Learning Rate: {current_lr:.7f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
        print(f"\n💾 ✅ New best model saved! (Val Acc: {best_val_acc*100:.2f}%)")
        
        # Print per-class accuracy
        print("\n📈 Per-class validation accuracy:")
        for i, cls in enumerate(classes):
            cls_mask = np.array(val_labels) == i
            if cls_mask.sum() > 0:
                cls_acc = (np.array(val_preds)[cls_mask] == i).sum() / cls_mask.sum()
                print(f"  {cls}: {cls_acc*100:.2f}%")
    else:
        patience_counter += 1
        print(f"\n⏳ No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("\n⏹️ Early stopping triggered.")
            break

# --- Final Evaluation ---
print("\n" + "="*60)
print("🔍 Final Evaluation on Test Set")
print("="*60)

model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc, y_pred, y_true = validate(model, test_loader)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
print(f"✅ Best Validation Accuracy: {best_val_acc*100:.2f}%")

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

# Per-class accuracy
print("\n📈 Per-class Test Accuracy:")
for i, cls in enumerate(classes):
    cls_mask = np.array(y_true) == i
    if cls_mask.sum() > 0:
        cls_acc = (np.array(y_pred)[cls_mask] == i).sum() / cls_mask.sum()
        print(f"  {cls}: {cls_acc*100:.2f}% ({cls_mask.sum()} samples)")

print("\n✅ Training complete!")