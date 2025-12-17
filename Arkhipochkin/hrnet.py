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
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_dir": "train_images",
    "test_dir": "test_images",
    "csv_path": "train_solution.csv",
    "base_filters": 32
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
print(f" Запуск на: {CONFIG['device']}")
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

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class FusionModule(nn.Module):
    def __init__(self, high_c, low_c):
        super(FusionModule, self).__init__()
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_c, low_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(low_c),
            nn.ReLU(inplace=True)
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_c, high_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(high_c)
        )
    def forward(self, x_high, x_low):
        upsampled_low = F.interpolate(self.low_to_high(x_low), size=x_high.shape[2:], mode='bilinear', align_corners=True)
        out_high = x_high + upsampled_low
        downsampled_high = self.high_to_low(x_high)
        out_low = x_low + downsampled_high
        return out_high, out_low

class CustomHRNet(nn.Module):
    def __init__(self, base_channels=32):
        super(CustomHRNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1_high = BasicBlock(64, base_channels)
        self.transition1 = nn.Sequential(
            nn.Conv2d(64, base_channels * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.layer1_low = BasicBlock(base_channels * 2, base_channels * 2)

        self.stage2_high = nn.Sequential(BasicBlock(base_channels, base_channels),
                                         BasicBlock(base_channels, base_channels))

        self.stage2_low = nn.Sequential(BasicBlock(base_channels * 2, base_channels * 2),
                                        BasicBlock(base_channels * 2, base_channels * 2))

        self.fusion1 = FusionModule(base_channels, base_channels * 2)
        self.stage3_high = nn.Sequential(BasicBlock(base_channels, base_channels),
                                         BasicBlock(base_channels, base_channels))

        self.stage3_low = nn.Sequential(BasicBlock(base_channels * 2, base_channels * 2),
                                        BasicBlock(base_channels * 2, base_channels * 2))

        self.fusion2 = FusionModule(base_channels, base_channels * 2)
        self.high_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.low_pool = nn.AdaptiveAvgPool2d((1, 1))
        total_channels = base_channels + (base_channels * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x_high = self.layer1_high(x)
        x_low = self.transition1(x)
        x_low = self.layer1_low(x_low)
        x_high = self.stage2_high(x_high)
        x_low = self.stage2_low(x_low)
        x_high, x_low = self.fusion1(x_high, x_low)
        x_high = self.stage3_high(x_high)
        x_low = self.stage3_low(x_low)
        x_high, x_low = self.fusion2(x_high, x_low)
        y_high = self.high_pool(x_high)
        y_low = self.low_pool(x_low)
        out = torch.cat([y_high, y_low], dim=1)
        out = self.classifier(out)
        return out.squeeze(1)

model = CustomHRNet(base_channels=CONFIG["base_filters"]).to(CONFIG["device"])
if os.path.exists(CONFIG["csv_path"]):
    try:
        full_df = pd.read_csv(CONFIG["csv_path"], header=None)
        if not str(full_df.iloc[0, 0]).isdigit():
            full_df = pd.read_csv(CONFIG["csv_path"])
            full_df.columns = ['id', 'target']
        else:
            full_df.columns = ['id', 'target']

        print(f" CSV загружен. Строк: {len(full_df)}")
    except Exception as e:
        print(f" Ошибка чтения CSV: {e}")
        full_df = pd.DataFrame({'id': range(100), 'target': [0] * 80 + [1] * 20})
else:
    print(f" Файл {CONFIG['csv_path']} не найден!")
    full_df = pd.DataFrame({'id': range(100), 'target': [0] * 80 + [1] * 20})

train_df, val_df = train_test_split(full_df, test_size=CONFIG["val_split"], stratify=full_df['target'],
                                    random_state=CONFIG["seed"])
class_counts = train_df['target'].value_counts().sort_index()
if len(class_counts) < 2:
    class_weights = torch.tensor([1.0, 1.0])
else:
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
sample_weights = [class_weights[int(t)] for t in train_df['target']]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = DeepFakeDataset(train_df, CONFIG["train_dir"], transform=train_transforms)
val_dataset = DeepFakeDataset(val_df, CONFIG["train_dir"], transform=val_test_transforms)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler,
                          num_workers=CONFIG["num_workers"])

val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

criterion = nn.BCEWithLogitsLoss()
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
    val_recall = recall_score(all_labels, preds, zero_division=0)
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
        tqdm.write(f" New Best Model Saved! Threshold: {best_threshold:.2f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend();
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], label='Val F1', color='green')
plt.title('F1 Score')
plt.legend();
plt.grid(True)
plt.savefig('training_metrics.png')
print("\n Графики сохранены.")
print("\n Генерация submission.csv...")

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
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,num_workers=CONFIG["num_workers"])
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
    print("Папка теста не найдена.")
