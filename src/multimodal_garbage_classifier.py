import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image Transformations
# Resize to 224x224 for ResNet50 compatibility
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Multimodal Dataset Class
class GarbageMultimodalDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None):
        """
        Args:
            root_dir: Base directory of garbage_data
            split: 'Train', 'Val', or 'Test'
            transform: PyTorch transforms for images
        """
        self.transform = transform
        # Target directory: e.g., /home/yikai.chen/TL/garbage_data/CVPR_2024_dataset_Train
        self.target_dir = os.path.join(root_dir, f'CVPR_2024_dataset_{split}')
        self.classes = ['Black', 'Blue', 'Green', 'TTR'] # TTR maps to "Other" in proposal
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load file paths and extract text from filenames
        for label, cls_name in enumerate(self.classes):
            cls_path = os.path.join(self.target_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_full_path = os.path.join(cls_path, img_name)
                    # Extract description: "greasy_pizza_box_1" becomes "greasy pizza box"
                    description = img_name.split('.')[0]
                    description = ''.join([i for i in description if not i.isdigit()]).replace('_', ' ').strip()
                    if not description: 
                        description = "garbage item"
                    self.data.append((img_full_path, description, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, desc, label = self.data[idx]
        
        # Image processing
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Text processing (BERT Tokenization)
        tokens = self.tokenizer(desc, padding='max_length', max_length=16, 
                                truncation=True, return_tensors="pt")
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize DataLoaders
base_path = './garbage_data' 

train_dataset = GarbageMultimodalDataset(base_path, split='Train', transform=train_transform)
val_dataset = GarbageMultimodalDataset(base_path, split='Val', transform=val_test_transform)
test_dataset = GarbageMultimodalDataset(base_path, split='Test', transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(train_dataset)} training samples.")

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalClassifier, self).__init__()
        
        # Image branch: ResNet50
        self.resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer to get a 2048-d feature vector
        in_features_img = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 
        
        # Text branch: BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        in_features_text = 768 # BERT base hidden size
        
        # Multimodal Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features_img + in_features_text, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract features from image
        img_features = self.resnet(image) # [Batch, 2048]
        
        # Extract features from text
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output # [Batch, 768]
        
        # Concatenate image and text features
        combined_features = torch.cat((img_features, text_features), dim=1)
        
        # Final prediction
        return self.classifier(combined_features)

# Initialize model
model = MultimodalClassifier(num_classes=4).to(device)

import torch.optim as optim
from tqdm import tqdm # For progress bar

# Hyperparameters, Loss, and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # Use a smaller LR for BERT/ResNet
num_epochs = 10 

print("Starting Training...")

train_losses = []
val_accs = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    val_acc = 100 * val_correct / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
    train_losses.append(avg_loss)
    val_accs.append(val_acc)

print("Training Complete!")

model_path = "multimodal_garbage_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model weights saved to {model_path}")

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_curve.png')
plt.close()


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    incorrect_samples = [] # To store samples for "Error Analysis"

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Record incorrect cases for visualization
            mask = predicted != labels
            if mask.any():
                # We save the image, the true label, and the prediction
                for i in range(len(mask)):
                    if mask[i] and len(incorrect_samples) < 8: # Just keep first 8 errors
                        incorrect_samples.append({
                            'img': images[i].cpu(),
                            'label': labels[i].cpu().item(),
                            'pred': predicted[i].cpu().item()
                        })

    # Print Classification Report
    print("\n--- Final Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Show Incorrect Classifications
    print("\n--- Incorrect Classifications Analysis ---")
    plt.figure(figsize=(15, 8))
    for i, sample in enumerate(incorrect_samples):
        plt.subplot(2, 4, i + 1)
        # Unnormalize image for display
        img = sample['img'].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"True: {test_dataset.classes[sample['label']]}\nPred: {test_dataset.classes[sample['pred']]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('error_analysis.png')

# Run Evaluation
evaluate_model(model, test_loader)
