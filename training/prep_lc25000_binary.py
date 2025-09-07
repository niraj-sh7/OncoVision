# training/prep_lc25000_binary.py
import shutil, random, os, pathlib, sys, glob

random.seed(1337)

DATA = pathlib.Path("data")
OUT = DATA / "lc25000_binary"

def find_root():
    """
    Look for a folder that contains 'colon_image_sets' and 'lung_image_sets'
    under data/. Handles common extract names:
      - data/lung_colon_image_set/...
      - data/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/...
      - or any nested variant.
    """
    # 1) direct guess
    candidates = [
        DATA / "lung_colon_image_set",
        DATA / "lung-and-colon-cancer-histopathological-images" / "lung_colon_image_set",
    ]
    for c in candidates:
        if (c / "colon_image_sets").exists() and (c / "lung_image_sets").exists():
            return c

    # 2) recursive search for a folder that has both subfolders
    for p in DATA.rglob("*"):
        if p.is_dir():
            if (p / "colon_image_sets").exists() and (p / "lung_image_sets").exists():
                return p
    return None

def collect_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return files

def main(limit_per_class=None):
    root = find_root()
    if not root:
        print("ERROR: Could not find LC25000 root with 'colon_image_sets' and 'lung_image_sets'.", file=sys.stderr)
        print("Check your unzip location under 'data/'", file=sys.stderr)
        sys.exit(1)

    colon = root / "colon_image_sets"
    lung  = root / "lung_image_sets"

    malignant_folders = [
        colon / "colon_aca",
        lung  / "lung_aca",
        lung  / "lung_scc",
    ]
    benign_folders = [
        colon / "colon_n",
        lung  / "lung_n",
    ]

    # sanity
    for f in malignant_folders + benign_folders:
        if not f.exists():
            print(f"WARNING: Expected folder missing: {f}")

    mal = []
    ben = []
    for f in malignant_folders:
        mal.extend(collect_images(f))
    for f in benign_folders:
        ben.extend(collect_images(f))

    if limit_per_class:
        mal = mal[:limit_per_class]
        ben = ben[:limit_per_class]

    print(f"Found malignant: {len(mal)} | benign: {len(ben)}")

    # build output tree
    for split in ["train", "val"]:
        for cls in ["cancer", "normal"]:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    def split(lst, frac=0.9):
        random.shuffle(lst)
        n = int(len(lst) * frac)
        return lst[:n], lst[n:]

    mal_train, mal_val = split(mal, 0.9)
    ben_train, ben_val = split(ben, 0.9)

    def copy_all(paths, dest_dir):
        for p in paths:
            shutil.copy2(p, dest_dir / p.name)

    copy_all(mal_train, OUT / "train" / "cancer")
    copy_all(ben_train, OUT / "train" / "normal")
    copy_all(mal_val,   OUT / "val"   / "cancer")
    copy_all(ben_val,   OUT / "val"   / "normal")

    print("Done. Output at:", OUT)
    for d in (OUT/"train"/"cancer", OUT/"train"/"normal", OUT/"val"/"cancer", OUT/"val"/"normal"):
        print(d, "->", len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) + len(list(d.glob("*.png"))), "images")

if __name__ == "__main__":
    # Optional: pass a small limit to test quickly, e.g., python training/prep_lc25000_binary.py 500
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit_per_class=lim)
