"""
MNIST Classification with ANN (Artificial Neural Network)
Basic feedforward network without validation set
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from utils.visualization import plot_classification_history, visualize_sample_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class MNISTNet_ANN(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MNISTNet_ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.001, 0.001)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


def prepare_data():
    """Prepare MNIST train and test datasets"""
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(loader)}, '
                  f'Loss: {loss.item():.6f}, Accuracy: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_loss, epoch_accuracy


def evaluate_model(model, loader, criterion):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def save_model(model, filepath='mnist_ann_model.pth'):
    """Save model checkpoint"""
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')

def main():
    print("="*60)
    print("MNIST Classification with ANN")
    print("="*60)
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, test_loader = prepare_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating ANN model...")
    model = MNISTNet_ANN().to(device)
    
    print("\nModel Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 15
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, num_epochs
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Test
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        history['test_acc'].append(test_acc)
        
        print(f'\nEpoch [{epoch}/{num_epochs}] Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Acc: {test_acc:.2f}%')
        print("-" * 60)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Test Results")
    print("="*60)
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    
    # Visualize training history
    print("\nGenerating training history plots...")
    plot_classification_history(history, "ANN - MNIST Training")
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    visualize_sample_predictions(model, test_loader, class_names, num_samples=10, device=device)
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
