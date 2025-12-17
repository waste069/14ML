import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

CONFIG = {
    "seed": 42,
    "img_size": 256,
    "batch_size": 32,  # Если будет ошибка CUDA out of memory, уменьшите до 16
    "epochs": 15,
    "lr": 0.001,
    "val_split": 0.2,
    "num_workers": 0,
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
if os.path.exists(CONFIG["csv_path"]):
    try:
        full_df = pd.read_csv(CONFIG["csv_path"], header=None)
        if not str(full_df.iloc[0, 0]).isdigit():
            full_df = pd.read_csv(CONFIG["csv_path"])

        full_df.columns = ['id', 'target']
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        full_df = pd.DataFrame({'id': range(100), 'target': [0] * 50 + [1] * 50})

    print(f" Данные загружены. Всего строк: {len(full_df)}")
else:
    print("заглушка")
    full_df = pd.DataFrame({'id': range(100), 'target': [0] * 80 + [1] * 20})


train_df, val_df = train_test_split(
    full_df,
    test_size=CONFIG["val_split"],
    stratify=full_df['target'],
    random_state=CONFIG["seed"]
)
train_dataset = DeepFakeDataset(train_df, CONFIG["train_dir"], transform=train_transforms)
val_dataset = DeepFakeDataset(val_df, CONFIG["train_dir"], transform=val_test_transforms)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

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

class CustomDeepFakeModel(nn.Module):
    def __init__(self):
        super(CustomDeepFakeModel, self).__init__()
        # Stride=2 вместо MaxPool для сохранения шума
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x

model = CustomDeepFakeModel().to(CONFIG["device"])
num_pos = full_df['target'].sum()
num_neg = len(full_df) - num_pos
pos_weight = torch.tensor([num_neg / max(num_pos, 1)]).to(CONFIG["device"])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_f1 = 0.0

print("\n обучение..")
for epoch in range(CONFIG["epochs"]):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]", leave=False)

    for images, labels in train_loop:
        images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]", leave=False)
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)
    scheduler.step(val_f1)
    epoch_time = time.time() - start_time
    lr = optimizer.param_groups[0]['lr']
    tqdm.write(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | "
               f"F1={val_f1:.4f} (Rec={val_recall:.2f}, Prec={val_precision:.2f}) | LR={lr:.6f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        tqdm.write(" New Best Model Saved!")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], label='Val F1 Score', color='green')
plt.title('F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.grid(True)

plt.savefig('training_metrics.png')
print("\n Графики сохранены в 'training_metrics.png'")
print("\nГен файла submission.csv..")
if os.path.exists(CONFIG["test_dir"]):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_files = [f for f in os.listdir(CONFIG["test_dir"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_data = []
    for f in test_files:
        try:
            tid = int(os.path.splitext(f)[0])
            test_data.append(tid)
        except ValueError:
            continue
    test_df = pd.DataFrame({'id': test_data}).sort_values('id')
    test_dataset = DeepFakeDataset(test_df, CONFIG["test_dir"], transform=val_test_transforms, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"])
    submission_ids = []
    submission_preds = []
    test_loop = tqdm(test_loader, desc="Predicting", leave=True)
    with torch.no_grad():
        for images, ids in test_loop:
            images = images.to(CONFIG["device"])
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            # Порог 0.5. Можно попробовать 0.45 для увеличения Recall!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            preds = (probs > 0.5).int().cpu().numpy()
            submission_ids.extend(ids.numpy())
            submission_preds.extend(preds)

    submission = pd.DataFrame({
        'Id': submission_ids,
        'target_feature': submission_preds
    })

    submission.to_csv('submission.csv', index=False)
    print(" Файл 'submission.csv' создан.")
else:
    print(f" Папка {CONFIG['test_dir']} не найдена. Пропуск этапа тестирования.")
