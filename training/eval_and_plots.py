import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

def load_model(weights_path, imagenet=False, device="cpu"):
    if imagenet:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)  
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.eval().to(device)
    return m

def infer_class_indices(class_to_idx: dict):
    """
    Try to detect which index corresponds to 'normal' vs 'cancer'.
    Accepts names like '0_normal', '1_cancer', 'normal', 'benign', 'cancer', case-insensitive.
    Returns (idx_normal, idx_cancer). If not found, returns (None, None).
    """
    idx_normal = None
    idx_cancer = None
    for name, idx in class_to_idx.items():
        key = name.lower()
        if ("normal" in key) or ("benign" in key):
            if key.startswith("0_") or key == "0_normal":
                idx_normal = idx
            elif idx_normal is None:
                idx_normal = idx
        if "cancer" in key:
            if key.startswith("1_") or key == "1_cancer":
                idx_cancer = idx
            elif idx_cancer is None:
                idx_cancer = idx
    return idx_normal, idx_cancer

def compute_channel_stats_from_train(train_dir):
    """
    Compute mean and std of training data.
    Returns (mean, std) as lists suitable for transforms.Normalize.
    """
    print("[eval] Computing normalization statistics from TRAINING data only...")
    temp_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=temp_tfm)
    loader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    for images, _ in loader:
        batch_size = images.shape[0]
        images_flat = images.view(images.shape[0], images.shape[1], -1)  # (batch, 3, H*W)
        mean += images_flat.mean(dim=2).sum(dim=0)
        std += images_flat.std(dim=2).sum(dim=0)
        total_pixels += batch_size
    
    mean /= total_pixels
    std /= total_pixels
    
    print(f"[eval] Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="e.g., data/lc25000_3way")
    ap.add_argument("--weights", default="artifacts/resnet18_histopath.pt")
    ap.add_argument("--imagenet", action="store_true")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prob_is_normal", type=int, choices=[0,1], default=None)
    ap.add_argument("--eval_split", choices=["val", "test"], default="test", 
                    help="Which split to evaluate on (test is final metrics)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute normalization statistics from TRAINING data only
    train_dir = os.path.join(args.data_dir, "train")
    mean, std = compute_channel_stats_from_train(train_dir)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load evaluation dataset (default: test set for final metrics)
    eval_root = os.path.join(args.data_dir, args.eval_split)
    eval_ds = datasets.ImageFolder(eval_root, transform=tfm)
    print(f"[eval] Loading {args.eval_split} set from {eval_root}")
    print(f"[eval] class_to_idx: {eval_ds.class_to_idx}")

    idx_normal, idx_cancer = infer_class_indices(eval_ds.class_to_idx)
    if idx_normal is None or idx_cancer is None:
        raise SystemExit(
            f"Could not infer class indices from {eval_ds.class_to_idx}. "
            "Make sure folders are named like '0_normal'/'1_cancer' or 'normal'/'cancer'."
        )

    if args.prob_is_normal is None:
        prob_is_normal = (idx_cancer == 0 and idx_normal == 1)
    else:
        prob_is_normal = bool(args.prob_is_normal)
    print(f"[eval] Treating model sigmoid as P({'normal' if prob_is_normal else 'cancer'})")

    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    model = load_model(args.weights, imagenet=args.imagenet, device=device)

    # Inference on evaluation set
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))             
            p_model = torch.sigmoid(logits).squeeze(1).cpu().numpy()  
            if prob_is_normal:
                p_cancer = 1.0 - p_model
            else:
                p_cancer = p_model
            y_true_all.extend(y.numpy().tolist())
            y_prob_all.extend(p_cancer.tolist())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)

    mask = (y_true == idx_normal) | (y_true == idx_cancer)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_true = (y_true == idx_cancer).astype(int)

    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = max(1, (tp + tn + fp + fn))
    acc = (tp + tn) / total
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_curve, prec_curve)

    summary = {
        "eval_split": args.eval_split,
        "eval_set_note": "TEST SET - Final held-out evaluation (no leakage)" if args.eval_split == "test" else "VALIDATION SET (for hyperparameter tuning)",
        "normalization_computed_from": "training set only",
        "acc": acc, "precision": prec, "recall": rec,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "class_to_idx": eval_ds.class_to_idx,
        "idx_normal": int(idx_normal),
        "idx_cancer": int(idx_cancer),
        "prob_is_normal": bool(prob_is_normal),
        "n_eval": int(total),
        "normalization_mean": mean,
        "normalization_std": std,
    }
    
    out_name = f"metrics_summary_{args.eval_split}.json"
    with open(os.path.join(args.out_dir, out_name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved metrics_summary_{args.eval_split}.json:")
    print(json.dumps(summary, indent=2))

    # Generate plots
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({args.eval_split})")
    plt.xticks([0, 1], ["Benign (0)", "Cancer (1)"])
    plt.yticks([0, 1], ["Benign (0)", "Cancer (1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center', color='black', fontsize=12)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, f"confusion_matrix_{args.eval_split}.png")
    plt.savefig(cm_path, dpi=220); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({args.eval_split})"); plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.out_dir, f"roc_curve_{args.eval_split}.png")
    plt.savefig(roc_path, dpi=220); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(rec_curve, prec_curve, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve ({args.eval_split})"); plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(args.out_dir, f"pr_curve_{args.eval_split}.png")
    plt.savefig(pr_path, dpi=220); plt.close()

    print(f"\n✓ Saved plots:")
    print(f"  {cm_path}")
    print(f"  {roc_path}")
    print(f"  {pr_path}")

if __name__ == "__main__":
    main()
