import argparse, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="artifacts/resnet18_histopath.pt")
    return ap.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=tfm)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x,y in train_dl:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # val
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x,y in val_dl:
                x, y = x.to(device), y.to(device)
                prob = torch.sigmoid(model(x)).squeeze(1)
                pred = (prob > 0.5).long()
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct/total if total else 0
        print(f"Epoch {epoch+1}/{args.epochs} — val acc: {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
            print(f"✔ Saved best weights to {args.out}")
    print("Done. Best val acc:", best)

if __name__ == "__main__":
    main()
