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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="e.g., data/lc25000_binary")
    ap.add_argument("--weights", default="artifacts/resnet18_histopath.pt")
    ap.add_argument("--imagenet", action="store_true")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prob_is_normal", type=int, choices=[0,1], default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_root = os.path.join(args.data_dir, "val")
    val = datasets.ImageFolder(val_root, transform=tfm)
    print("class_to_idx:", val.class_to_idx)

    idx_normal, idx_cancer = infer_class_indices(val.class_to_idx)
    if idx_normal is None or idx_cancer is None:
        raise SystemExit(
            f"Could not infer class indices from {val.class_to_idx}. "
            "Make sure folders are named like '0_normal'/'1_cancer' or 'normal'/'cancer'."
        )

    if args.prob_is_normal is None:
        prob_is_normal = (idx_cancer == 0 and idx_normal == 1)
    else:
        prob_is_normal = bool(args.prob_is_normal)
    print(f"[eval] Treating model sigmoid as P({'normal' if prob_is_normal else 'cancer'})")

    loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    model = load_model(args.weights, imagenet=args.imagenet, device=device)

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
        "acc": acc, "precision": prec, "recall": rec,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "class_to_idx": val.class_to_idx,
        "idx_normal": int(idx_normal),
        "idx_cancer": int(idx_cancer),
        "prob_is_normal": bool(prob_is_normal),
        "n_eval": int(total)
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved metrics_summary.json:", summary)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Benign (0)", "Cancer (1)"])
    plt.yticks([0, 1], ["Benign (0)", "Cancer (1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center', color='black', fontsize=12)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=220); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--', alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=220); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(rec_curve, prec_curve, label=f"AUC = {pr_auc:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve"); plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(args.out_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=220); plt.close()

    print("Saved:", cm_path, roc_path, pr_path)

if __name__ == "__main__":
    main()
