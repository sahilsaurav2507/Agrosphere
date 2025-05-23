import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import traceback

# Paths
train_dir = r"E:\Agrospere\models\datas\crop_health_daata\New Plant Diseases Dataset(Augmented)\train"
valid_dir = r"E:\Agrospere\models\datas\crop_health_daata\New Plant Diseases Dataset(Augmented)\valid"
test_dir = r"E:\Agrospere\models\datas\crop_health_daata\test"






# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")















# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets



try:
    train_dataset = ImageFolder(train_dir, transform=transform)
    valid_dataset = ImageFolder(valid_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)
except Exception as e:
    print(f"Error loading datasets: {e}")
    traceback.print_exc()
    exit(1)





# Create data loaders with reduced workers
try:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
except Exception as e:
    print(f"Error creating data loaders: {e}")
    traceback.print_exc()
    exit(1)

# Get number of classes
num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")



# CNN Model
class CropHealthCNN(nn.Module):
    def __init__(self, num_classes):








        super(CropHealthCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        








        # Calculate the correct input size for the classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),




            nn.Linear(512, num_classes)
        )



    















    def forward(self, x):







        x = self.features(x)
        x = self.classifier(x)
        return x























# Initialize model, loss function, and optimizer




try:
    model = CropHealthCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
except Exception as e:
    print(f"Error initializing model: {e}")
    traceback.print_exc()
    exit(1)

# TensorBoard setup

writer = SummaryWriter('runs/crop_health_experiment')


# Training function
def train_model():

    best_valid_acc = 0.0
    









    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            try:
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
                


                # Print progress every 100 batches
                if (i+1) % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            











            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                traceback.print_exc()
                continue
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()




























                except Exception as e:
                    print(f"Error in validation: {e}")
                    traceback.print_exc()
                    continue
        
        val_loss = val_loss / len(valid_loader)
        val_acc = 100 * val_correct / val_total
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            try:
                torch.save(model.state_dict(), 'crop_health_best_model.pth')
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')








            except Exception as e:
                print(f"Error saving model: {e}")
                traceback.print_exc()

# Test function
def test_model():
    try:



        # Load best model
        model.load_state_dict(torch.load('crop_health_best_model.pth'))
        model.eval()
        


        test_correct = 0
        test_total = 0


        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)



                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()






        

        test_acc = 100 * test_correct / test_total

        print(f'Test Accuracy: {test_acc:.2f}%')
        
        # Log class names to TensorBoard
        class_names = train_dataset.classes






        writer.add_text('Classes', str(class_names))



    except Exception as e:

        print(f"Error in testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:



        # Skip TensorBoard graph visualization which might be causing issues
        # Train the model
        print("Starting training...")
        train_model()
        

        # Test the model
        print("Starting testing...")
        test_model()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
    finally:

        # Close TensorBoard writer
        writer.close()
        print("Training completed.")
