import os
import sys
import argparse
import random
from pathlib import Path
from collections import Counter
import time
import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# утилиты и настройки воспроизводимости


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# датасет


class FacesDataset(Dataset):
    def __init__(self, df, images_dir, mode='train', transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{int(row['Id'])}.jpg"
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.mode == 'test':
            return img, int(row['Id'])
        else:
            label = int(row['target_feature'])
            return img, label


# сама архитектура


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class DeepFakeNet(nn.Module):
    def __init__(self, num_classes=1, base_filters=32, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.SiLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, blocks=3, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, blocks=3, stride=2)

        self.context_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear((base_filters*8) + (base_filters*8), 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers = [ConvBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)

        pooled = self.context_pool(feat_map)
        pooled = pooled.view(pooled.size(0), -1)

        gap = F.adaptive_avg_pool2d(feat_map, (1,1)).view(feat_map.size(0), -1)

        cat = torch.cat([pooled, gap], dim=1)
        out = self.head(cat)
        return out.squeeze(1)


# метрики


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# тренировка


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.extend(probs.tolist())
        targets.extend(lbls.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(targets)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            logits = model(imgs)
            loss = criterion(logits, lbls.float())
            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            targets.extend(lbls.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(targets)

# вспомогательные функции

def compute_metrics_at_threshold(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return prec, rec, f1


def find_best_threshold(y_true, y_probs, target='f1'):
    # ищем порог, который оптимизирует f1, но отдаём приоритет recall при равных f1
    best_t = 0.5
    best_metric = -1
    best_rec = 0
    for t in np.linspace(0.1, 0.95, 85):
        prec, rec, f1 = compute_metrics_at_threshold(y_true, y_probs, t)
        metric = f1
        if metric > best_metric or (math.isclose(metric, best_metric) and rec > best_rec):
            best_metric = metric
            best_t = t
            best_rec = rec
    return best_t, best_metric, best_rec


def build_transforms(image_size=256, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])


def prepare_dataframes(data_dir):
    # Загружаем train_solution
    sol = pd.read_csv(os.path.join(data_dir, 'train_solution.csv'))
    sol.columns = ['Id', 'target_feature']

    # Создадим train/val split по пациентам (Id-based split here)
    train_df, val_df = train_test_split(sol, test_size=0.12, stratify=sol['target_feature'], random_state=42)
    test_ids = []
    # Сформируем test_df шаблон (Id column required for dataset loader)
    test_dir = os.path.join(data_dir, 'test_images')
    # Предполагаем последовательные номера 0..N-1, но безопаснее — прочитать файлы в папке
    test_files = sorted([p for p in os.listdir(test_dir) if p.lower().endswith(('.jpg','.png','.jpeg'))])
    ids = []
    for fn in test_files:
        stem = os.path.splitext(fn)[0]
        try:
            ids.append(int(stem))
        except:
            # если имена нестандартные, создадим псевдо-Id
            continue
    test_df = pd.DataFrame({'Id': ids})
    return train_df, val_df, test_df


def plot_metrics(history, out_dir):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_f1'], label='train_f1')
    plt.plot(epochs, history['val_f1'], label='val_f1')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'f1.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_rec'], label='train_rec')
    plt.plot(epochs, history['val_rec'], label='val_rec')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'recall.png'))
    plt.close()




def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    os.makedirs(args.out_dir, exist_ok=True)

    print('Подготовка данных....')
    train_df, val_df, test_df = prepare_dataframes(args.data_dir)
    print('Train size:', len(train_df), 'Val size:', len(val_df), 'Test size:', len(test_df))

    train_trans = build_transforms(256, 'train')
    val_trans = build_transforms(256, 'val')

    train_ds = FacesDataset(train_df, os.path.join(args.data_dir, 'train_images'), mode='train', transform=train_trans)
    val_ds = FacesDataset(val_df, os.path.join(args.data_dir, 'train_images'), mode='train', transform=val_trans)

    # учёт дисбаланса: взвешенный сэмплинг
    counter = Counter(train_df['target_feature'].tolist())
    print('Class counts:', counter)
    class_sample_count = [counter.get(0,0), counter.get(1,0)]
    weights = 1. / torch.tensor([class_sample_count[c] for c in train_df['target_feature']], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepFakeNet(num_classes=1, base_filters=args.base_filters, dropout=args.dropout).to(device)

    total = sum(class_sample_count)
    weight_for_0 = total / (2 * class_sample_count[0])
    weight_for_1 = total / (2 * class_sample_count[1])
    bce_weights = torch.tensor([weight_for_0, weight_for_1], device=device)

    pos_weight = torch.tensor([weight_for_1 / weight_for_0], device=device)
    bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal = FocalLoss(alpha=1.0, gamma=2.0)

    def combined_loss(logits, targets):
        return 0.6 * bce_with_logits(logits, targets.float()) + 0.4 * focal(logits, targets.float())

    # оптимизатор и планировщик
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # тренировка
    best_val_f1 = -1
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')
    history = {'train_loss':[], 'val_loss':[], 'train_f1':[], 'val_f1':[], 'train_rec':[], 'val_rec':[]}

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_probs, train_targets = train_one_epoch(model, train_loader, optimizer, combined_loss, device)
        val_loss, val_probs, val_targets = validate(model, val_loader, combined_loss, device)

        # порог на валидации
        best_t, best_f1, best_rec = find_best_threshold(val_targets, val_probs)
        # метрики при 0.5
        tr_prec, tr_rec, tr_f1 = compute_metrics_at_threshold(train_targets, train_probs, threshold=best_t)
        val_prec, val_rec, val_f1 = compute_metrics_at_threshold(val_targets, val_probs, threshold=best_t)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1)
        history['train_rec'].append(tr_rec)
        history['val_rec'].append(val_rec)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict(), 'threshold': best_t, 'epoch': epoch}, best_model_path)
            print('best модель была сохранена (f1=%.4f, rec=%.4f, t=%.3f)'.format() % (val_f1, val_rec, best_t))

        scheduler.step(val_f1)

        print(f'Epoch {epoch}/{args.epochs} | time {time.time()-t0:.1f}s')
        print(f'  train_loss {train_loss:.4f} val_loss {val_loss:.4f}')
        print(f'  train_f1 {tr_f1:.4f} val_f1 {val_f1:.4f} (thr {best_t:.3f})')
        print(f'  train_recall {tr_rec:.4f} val_recall {val_rec:.4f}')

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    plot_metrics(history, args.out_dir)

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    thr = float(ckpt.get('threshold', 0.5))
    print('наиучший порог из валидации:', thr)

    test_trans = build_transforms(256, 'val')
    test_ds = FacesDataset(test_df, os.path.join(args.data_dir, 'test_images'), mode='test', transform=test_trans)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for imgs, batch_ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
            ids.extend([int(x) for x in batch_ids])

    preds = np.array(preds)
    ids = np.array(ids)
    final_preds = (preds >= thr).astype(int)

    sub = pd.DataFrame({'Id': ids, 'target_feature': final_preds})
    sub = sub.sort_values('Id')
    sub.to_csv(os.path.join(args.out_dir, 'submission.csv'), index=False)
    print('submission сохранен в', os.path.join(args.out_dir, 'submission.csv'))

