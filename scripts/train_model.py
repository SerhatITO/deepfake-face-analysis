import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ====================
# 📁 Dizinler ve Ayarlar
# ====================
DATA_DIR = "../faces"  # Real ve fake yüzlerin olduğu klasör
BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../models/deepfake_resnet18.pth"

# ====================
# 🔄 Transformlar
# ====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ====================
# 🔍 Dataset & DataLoader
# ====================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# 🧠 Model (ResNet18)
# ====================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 sınıf: real & fake
model = model.to(DEVICE)

# ====================
# ⚙️ Loss ve Optimizer
# ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====================
# 🎯 Eğitim Döngüsü
# ====================
print("\n🚀 Eğitim Başladı...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    print(f"\n📊 Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")

# ====================
# 💾 Modeli Kaydet
# ====================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Model kaydedildi: {MODEL_PATH}")
