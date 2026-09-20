import pickle, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"e:/PreserveNet/PreserveNet")
FAKE_ROOT = REPO_ROOT / "fake_cifar10"
BATCH_DIR = FAKE_ROOT / "cifar10" / "cifar-10-batches-py"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

def make_batch(n, name):
    labels = [i % 10 for i in range(n)]
    data = RNG.integers(0, 256, size=(n, 3*32*32), dtype=np.uint8)
    (BATCH_DIR / name).write_bytes(
        pickle.dumps({b"labels": labels, b"data": data}, protocol=2)
    )

for i in range(1, 6):
    make_batch(40, f"data_batch_{i}")
make_batch(40, "test_batch")
(BATCH_DIR / "batches.meta").write_bytes(pickle.dumps({
    b"label_names": [
        b"airplane", b"automobile", b"bird", b"cat", b"deer",
        b"dog", b"frog", b"horse", b"ship", b"truck",
    ]
}, protocol=2))
print("Fake CIFAR-10 written.")

# Bypass torchvision MD5 check (fake data won't match real checksums)
import torchvision.datasets.cifar as _tv
_tv.CIFAR10._check_integrity = lambda self: True

sys.path.insert(0, str(REPO_ROOT))
from src.data.datasets import get_cifar10_loaders, CIFAR10_CLASSES
from src.data.transforms import denormalise
import torch

IMAGE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
print("image_size =", IMAGE_SIZE)

t0 = time.perf_counter()
train_loader, val_loader, meta = get_cifar10_loaders(
    data_dir=FAKE_ROOT,
    batch_size=32,
    image_size=IMAGE_SIZE,
    num_workers=0,
    augment=True,
    download=False,
)
print("Loaders built in", round(time.perf_counter() - t0, 3), "s")
print("Meta         :", meta)
print("Train batches:", len(train_loader))
print("Val   batches:", len(val_loader))

imgs, labels = next(iter(train_loader))
print("\nFirst train batch")
print("  imgs  shape:", tuple(imgs.shape), "dtype =", imgs.dtype)
print("  labels shape:", tuple(labels.shape))
print("  pixel range: [", round(imgs.min().item(), 4), ",", round(imgs.max().item(), 4), "]")

counts = {}
for lbl in labels.tolist():
    k = CIFAR10_CLASSES[lbl]
    counts[k] = counts.get(k, 0) + 1
print("  label counts:", counts)

assert imgs.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE), f"Bad shape: {imgs.shape}"
assert imgs.dtype == torch.float32, f"Bad dtype: {imgs.dtype}"

v_imgs, v_labels = next(iter(val_loader))
print("\nFirst val batch")
print("  imgs  shape:", tuple(v_imgs.shape))
print("  pixel range: [", round(v_imgs.min().item(), 4), ",", round(v_imgs.max().item(), 4), "]")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seen = {}
for bi, bl in train_loader:
    for img, lbl in zip(bi, bl):
        li = int(lbl)
        if li not in seen:
            seen[li] = img
        if len(seen) == 10:
            break
    if len(seen) == 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle(
    "CIFAR-10 smoke test  image_size=" + str(IMAGE_SIZE),
    fontsize=12, fontweight="bold", color="#e8e8f8",
)
for cls_idx in range(10):
    ax = axes[cls_idx // 5, cls_idx % 5]
    img = denormalise(seen[cls_idx], mean=meta.mean, std=meta.std)
    ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_title(CIFAR10_CLASSES[cls_idx], fontsize=8.5, color="#e0e0ff")
    ax.axis("off")

plt.tight_layout()
out_png = REPO_ROOT / ("cifar10_smoke_" + str(IMAGE_SIZE) + ".png")
plt.savefig(str(out_png), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nImage grid saved:", out_png)
print("\nSMOKE TEST PASSED")
print("  imgs  :", tuple(imgs.shape))
print("  PNG   :", out_png)
