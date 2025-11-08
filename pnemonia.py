import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm
import time

# Fix OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ No GPU detected, using CPU")

# Configuration
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Set paths
BASE_DIR = r"D:\Technical Projects\SWHACK ganavi\pnemonia"
DATA_DIR = os.path.join(BASE_DIR, 'chest_xray')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_pneumonia_model.pth')
HISTORY_PLOT_PATH = os.path.join(BASE_DIR, 'training_history.png')

# Verify directories exist
print(f"Checking directories...")
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Train directory exists: {os.path.exists(os.path.join(DATA_DIR, 'train'))}")
print(f"Val directory exists: {os.path.exists(os.path.join(DATA_DIR, 'val'))}")
print(f"Test directory exists: {os.path.exists(os.path.join(DATA_DIR, 'test'))}")


class PneumoniaCNN(nn.Module):
    """CNN architecture for pneumonia detection"""
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fourth Convolutional Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Dense Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


class PneumoniaDetector:
    def __init__(self, img_size=IMG_SIZE, data_dir=DATA_DIR, model_save_path=MODEL_SAVE_PATH):
        self.img_size = img_size
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.model = None
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': []
        }
        
    def create_data_loaders(self):
        """Create data loaders with augmentation for training"""
        # Training data augmentation
        train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation and test data (only normalization)
        val_test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=val_test_transform
        )
        
        test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=val_test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader, train_dataset.class_to_idx
    
    def build_model(self):
        """Build CNN model"""
        self.model = PneumoniaCNN().to(self.device)
        return self.model
    
    def calculate_metrics(self, outputs, labels):
        """Calculate precision and recall"""
        predictions = (outputs > 0.5).float()
        
        # True Positives, False Positives, False Negatives
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall
    
    def train_epoch(self, loader, criterion, optimizer, epoch_num, total_epochs):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_precisions = []
        all_recalls = []
        
        # Progress bar for batches
        pbar = tqdm(loader, desc=f'Epoch {epoch_num}/{total_epochs} [TRAIN]', 
                    leave=False, ncols=120, colour='green')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.float().to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            precision, recall = self.calculate_metrics(outputs, labels)
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            # Update progress bar
            current_acc = correct / total
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'GPU_MB': f'{torch.cuda.memory_allocated()/1024**2:.0f}' if torch.cuda.is_available() else 'N/A'
            })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        epoch_precision = np.mean(all_precisions)
        epoch_recall = np.mean(all_recalls)
        
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall
    
    def validate_epoch(self, loader, criterion, epoch_num, total_epochs):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_precisions = []
        all_recalls = []
        
        # Progress bar for validation
        pbar = tqdm(loader, desc=f'Epoch {epoch_num}/{total_epochs} [VAL]', 
                    leave=False, ncols=120, colour='blue')
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                precision, recall = self.calculate_metrics(outputs, labels)
                all_precisions.append(precision)
                all_recalls.append(recall)
                
                # Update progress bar
                current_acc = correct / total
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        epoch_precision = np.mean(all_precisions)
        epoch_recall = np.mean(all_recalls)
        
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall
    
    def train(self, train_loader, val_loader, epochs=EPOCHS):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Print model summary
        print(self.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, 
            min_lr=1e-7, verbose=True
        )
        
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        print("\n" + "="*80)
        print("TRAINING STARTED".center(80))
        print("="*80)
        
        start_time = time.time()
        
        # Main training loop with overall progress
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_prec, train_rec = self.train_epoch(
                train_loader, criterion, optimizer, epoch+1, epochs
            )
            
            # Validate
            val_loss, val_acc, val_prec, val_rec = self.validate_epoch(
                val_loader, criterion, epoch+1, epochs
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_precision'].append(train_prec)
            self.history['val_precision'].append(val_prec)
            self.history['train_recall'].append(train_rec)
            self.history['val_recall'].append(val_rec)
            
            # Calculate time metrics
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs
            
            # Print comprehensive epoch summary
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{epochs} SUMMARY".center(80))
            print(f"{'='*80}")
            print(f"⏱️  Epoch Time: {epoch_time:.2f}s | Elapsed: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min")
            print(f"\n📊 TRAINING:")
            print(f"   Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"   Precision: {train_prec:.4f} | Recall: {train_rec:.4f}")
            print(f"\n📈 VALIDATION:")
            print(f"   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"   Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_cached = torch.cuda.memory_reserved() / 1024**2
                print(f"\n💾 GPU Memory: {gpu_memory:.0f}MB allocated | {gpu_cached:.0f}MB cached")
            
            # Early stopping and model checkpointing
            if val_acc > best_val_acc:
                improvement = val_acc - best_val_acc
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"\n✅ NEW BEST MODEL SAVED!")
                print(f"   Path: {self.model_save_path}")
                print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
                print(f"   Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"\n⚠️  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED".center(80))
                    print(f"{'='*80}")
                    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
                    break
            
            print(f"{'='*80}\n")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE".center(80))
        print(f"{'='*80}")
        print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"💾 Model saved to: {self.model_save_path}")
        print(f"{'='*80}\n")
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        all_precisions = []
        all_recalls = []
        
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET".center(80))
        print("="*80)
        
        # Progress bar for testing
        pbar = tqdm(test_loader, desc='Testing', ncols=120, colour='cyan')
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                outputs = self.model(inputs).squeeze()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                precision, recall = self.calculate_metrics(outputs, labels)
                all_precisions.append(precision)
                all_recalls.append(recall)
                
                # Update progress
                pbar.set_postfix({'Acc': f'{correct/total:.4f}'})
        
        accuracy = correct / total
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print("\n" + "="*80)
        print("TEST SET RESULTS".center(80))
        print("="*80)
        print(f"🎯 ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎯 PRECISION: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        print(f"🎯 RECALL:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        print(f"🎯 F1-SCORE:  {f1_score:.4f} ({f1_score*100:.2f}%)")
        print("="*80 + "\n")
        
        return {
            'accuracy': accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }
    
    def plot_training_history(self):
        """Visualize training history"""
        if not self.history['train_loss']:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Accuracy
        axes[0, 0].plot(epochs_range, self.history['train_acc'], label='Train Accuracy')
        axes[0, 0].plot(epochs_range, self.history['val_acc'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(epochs_range, self.history['train_loss'], label='Train Loss')
        axes[0, 1].plot(epochs_range, self.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(epochs_range, self.history['train_precision'], label='Train Precision')
        axes[1, 0].plot(epochs_range, self.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(epochs_range, self.history['train_recall'], label='Train Recall')
        axes[1, 1].plot(epochs_range, self.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(HISTORY_PLOT_PATH, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {HISTORY_PLOT_PATH}")
        plt.show()
    
    def predict_image(self, img_path):
        """Predict pneumonia for a single image"""
        self.model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor).item()
        
        result = {
            'prediction': 'PNEUMONIA' if prediction > 0.5 else 'NORMAL',
            'confidence': prediction if prediction > 0.5 else 1 - prediction,
            'raw_score': prediction
        }
        
        return result


# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = PneumoniaDetector()
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, class_to_idx = detector.create_data_loaders()
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class indices: {class_to_idx}")
    
    # Build and train model
    print("\nBuilding model...")
    detector.build_model()
    
    print("\nStarting training...")
    detector.train(train_loader, val_loader)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = detector.evaluate(test_loader)
    
    print(f"\n✅ All outputs saved to: {BASE_DIR}")
    print(f"   - Model: {MODEL_SAVE_PATH}")
    print(f"   - Training plot: {HISTORY_PLOT_PATH}")
    
    # Example prediction (uncomment and adjust path as needed)
    # test_image = os.path.join(DATA_DIR, 'test', 'PNEUMONIA', 'person1_virus_6.jpeg')
    # result = detector.predict_image(test_image)
    # print(f"\nPrediction: {result['prediction']}")
    # print(f"Confidence: {result['confidence']:.2%}")