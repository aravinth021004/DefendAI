"""
Template for training the hybrid deepfake detection model.
This is a basic structure - you'll need to adapt it for your specific dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from deepfake_detector import HybridDeepfakeDetector
from PIL import Image
import cv2

class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection.
    Expected structure:
    dataset/
    ├── real/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── fake/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    """
    
    def __init__(self, data_dir, transform=None, sequence_length=1):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.samples = []
        
        # Load real samples (label 0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(real_dir, filename), 0))
        
        # Load fake samples (label 1)
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(fake_dir, filename), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # For sequence modeling, we'll create a sequence of length 1 for images
        # For videos, you would load multiple frames here
        image = image.unsqueeze(0)  # Add sequence dimension (1, C, H, W)
        
        return image, torch.tensor(label, dtype=torch.long)

def train_model(data_dir, model_save_path, epochs=50, batch_size=16, learning_rate=0.001):
    """
    Train the hybrid deepfake detection model.
    
    Args:
        data_dir: Path to dataset directory
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=os.path.join(data_dir, 'train'),
        transform=transform
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=os.path.join(data_dir, 'val'),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = HybridDeepfakeDetector(feature_dim=512, num_classes=2)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        val_probabilities = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # For metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                val_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of fake class
        
        val_acc = 100. * val_correct / val_total
        
        # Calculate additional metrics
        precision = precision_score(val_targets, val_predictions)
        recall = recall_score(val_targets, val_predictions)
        f1 = f1_score(val_targets, val_predictions)
        auc = roc_auc_score(val_targets, val_probabilities)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_auc': auc,
            }, model_save_path)
            print(f'  New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
        print()
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

def evaluate_model(model_path, test_data_dir):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to the saved model
        test_data_dir: Path to test dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = HybridDeepfakeDetector(feature_dim=512, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test dataset
    test_dataset = DeepfakeDataset(
        data_dir=test_data_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluation
    test_predictions = []
    test_targets = []
    test_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
            test_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_targets, test_predictions)
    precision = precision_score(test_targets, test_predictions)
    recall = recall_score(test_targets, test_predictions)
    f1 = f1_score(test_targets, test_predictions)
    auc = roc_auc_score(test_targets, test_probabilities)
    
    print("Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    # Example usage
    data_directory = "path/to/your/dataset"  # Update this path
    model_save_path = "../models/hybrid_deepfake_model.pth"
    
    # Train the model
    print("Starting model training...")
    train_model(
        data_dir=data_directory,
        model_save_path=model_save_path,
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    # Evaluate the model (optional)
    test_data_directory = "path/to/your/test/dataset"  # Update this path
    if os.path.exists(test_data_directory):
        print("\nEvaluating model...")
        evaluate_model(model_save_path, test_data_directory)
