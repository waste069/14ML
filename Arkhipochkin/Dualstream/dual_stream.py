import os
import time
import random
import math
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

CONFIG = {
    "seed": 42,
    "img_size": 256,
    "batch_size": 32,
    "epochs": 15,
    "lr": 0.001,
    "val_split": 0.2,
    "num_workers": 0,  # 0 для Windows
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_dir": "train_images",
    "test_dir": "test_images",
    "csv_path": "train_solution.csv"
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(CONFIG["seed"])

print(f" Запуск Dual-Stream {CONFIG['device']}")

class DeepFakeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row.iloc[0])
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (256, 256))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, int(img_id)
        else:
            label = torch.tensor(row.iloc[1], dtype=torch.float32)
            return image, label

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DualStreamNet(nn.Module):
    def __init__(self):
        super(DualStreamNet, self).__init__()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.spatial_layer1 = ResidualBlock(64, 64, stride=1)
        self.spatial_layer2 = ResidualBlock(64, 128, stride=2)
        self.spatial_layer3 = ResidualBlock(128, 256, stride=2)
        self.spatial_layer4 = ResidualBlock(256, 512, stride=2)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.freq_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.freq_layer1 = ResidualBlock(32, 64, stride=2)
        self.freq_layer2 = ResidualBlock(64, 128, stride=2)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        s = self.spatial_stem(x)
        s = self.spatial_layer1(s)
        s = self.spatial_layer2(s)
        s = self.spatial_layer3(s)
        s = self.spatial_layer4(s)
        s_vec = self.spatial_pool(s)
        fft = torch.fft.rfft2(x, norm='ortho')
        fft_mag = torch.abs(fft)
        fft_log = torch.log(fft_mag + 1e-8)
        fft_img = F.interpolate(fft_log, size=(128, 128), mode='bilinear', align_corners=False)
        f = self.freq_stem(fft_img)
        f = self.freq_layer1(f)
        f = self.freq_layer2(f)
        f_vec = self.freq_pool(f)
        combined = torch.cat([s_vec, f_vec], dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)
model = DualStreamNet().to(CONFIG["device"])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()
if os.path.exists(CONFIG["csv_path"]):
    try:
        full_df = pd.read_csv(CONFIG["csv_path"], header=None)
        if not str(full_df.iloc[0, 0]).isdigit():
            full_df = pd.read_csv(CONFIG["csv_path"])
        full_df.columns = ['id', 'target']
        print(f"📊 CSV загружен: {len(full_df)} строк")
    except Exception as e:
        print(f"Ошибка CSV: {e}")
        full_df = pd.DataFrame({'id': range(100), 'target': [0] * 80 + [1] * 20})
else:
    print(f"Файл {CONFIG['csv_path']} не найден")
    full_df = pd.DataFrame({'id': range(100), 'target': [0] * 80 + [1] * 20})
train_df, val_df = train_test_split(full_df, test_size=CONFIG["val_split"], stratify=full_df['target'], random_state=CONFIG["seed"])
class_counts = train_df['target'].value_counts().sort_index()
if len(class_counts) < 2:
    class_weights = torch.tensor([1.0, 1.0])
else:
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
sample_weights = [class_weights[int(t)] for t in train_df['target']]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_dataset = DeepFakeDataset(train_df, CONFIG["train_dir"], transform=train_transforms)
val_dataset = DeepFakeDataset(val_df, CONFIG["train_dir"], transform=val_test_transforms)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=CONFIG["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def find_best_threshold(y_true, y_probs):
    best_t = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_val_f1 = 0.0
best_threshold = 0.5

print("\n Начинаем обучение...")

for epoch in range(CONFIG["epochs"]):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]", leave=False)
    for images, labels in train_loop:
        images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
    avg_train_loss = running_loss / len(train_loader)
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    cur_threshold, cur_f1 = find_best_threshold(np.array(all_labels), np.array(all_probs))
    preds = (np.array(all_probs) >= cur_threshold).astype(int)
    val_recall = recall_score(all_labels, preds)
    val_precision = precision_score(all_labels, preds, zero_division=0)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(cur_f1)
    scheduler.step(cur_f1)
    lr = optimizer.param_groups[0]['lr']
    tqdm.write(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
               f"F1={cur_f1:.4f} (Best Thr={cur_threshold:.2f}) | Rec={val_recall:.2f} | LR={lr:.6f}")

    if cur_f1 > best_val_f1:
        best_val_f1 = cur_f1
        best_threshold = cur_threshold
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': float(best_threshold)
        }, "best_model.pth")
        tqdm.write(f"New Best Model Saved Threshold: {best_threshold:.2f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], label='Val F1', color='green')
plt.title('F1 Score')
plt.legend(); plt.grid(True)
plt.savefig('training_metrics.png')
print("\n Графики сохранены.")


print("\n  submission.csv...")

if os.path.exists(CONFIG["test_dir"]):
    checkpoint = torch.load("best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_thr = checkpoint['threshold']
    print(f"Используем лучший порог из обучения: {final_thr:.3f}")
    model.eval()

    test_files = [f for f in os.listdir(CONFIG["test_dir"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_data = []
    for f in test_files:
        try:
            tid = int(os.path.splitext(f)[0])
            test_data.append(tid)
        except:
            continue

    test_df = pd.DataFrame({'id': test_data}).sort_values('id')
    test_dataset = DeepFakeDataset(test_df, CONFIG["test_dir"], transform=val_test_transforms, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    submission_ids = []
    submission_preds = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(CONFIG["device"])
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > final_thr).int().cpu().numpy()
            submission_ids.extend(ids.numpy())
            submission_preds.extend(preds)
    submission = pd.DataFrame({'Id': submission_ids, 'target_feature': submission_preds})
    submission.to_csv('submission.csv', index=False)
    print(" Готово!")
else:
    print(" Папка теста не найдена.")
