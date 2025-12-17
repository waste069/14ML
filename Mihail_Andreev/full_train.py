import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from deepfake_detector import (
    DeepFakeNet, FacesDataset, build_transforms,
    FocalLoss, train_one_epoch, set_seed
)


def fine_tune(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sol = pd.read_csv(os.path.join(args.data_dir, "train_solution.csv"))
    sol.columns = ["Id", "target_feature"]


    train_trans = build_transforms(256, "train")
    train_ds = FacesDataset(sol, os.path.join(args.data_dir, "train_images"),
                            mode="train", transform=train_trans)

    # баланс классов
    counter = Counter(sol["target_feature"].tolist())
    class_sample_count = [counter.get(0, 0), counter.get(1, 0)]
    print("Class counts:", counter)

    weights = 1. / torch.tensor(
        [class_sample_count[c] for c in sol["target_feature"]],
        dtype=torch.float
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)

    model = DeepFakeNet(num_classes=1, base_filters=args.base_filters, dropout=args.dropout).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    old_thr = ckpt.get("threshold", 0.5)

    print("Загружена модель. Старый порог:", old_thr)

    total = sum(class_sample_count)
    weight_for_0 = total / (2 * class_sample_count[0])
    weight_for_1 = total / (2 * class_sample_count[1])
    pos_weight = torch.tensor([weight_for_1 / weight_for_0], device=device)

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal = FocalLoss()

    def combined_loss(logits, targets):
        return 0.6 * bce(logits, targets.float()) + 0.4 * focal(logits, targets.float())

    # LR сильно уменьшаем!
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loss, probs, targets = train_one_epoch(model, train_loader, optimizer, combined_loss, device)
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f}")

    save_path = os.path.join(args.out_dir, "best_model_finetuned.pth")
    torch.save({"model_state_dict": model.state_dict(), "threshold": old_thr}, save_path)
    print("Модель сохранена в:", save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./finetuned")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fine_tune(args)
