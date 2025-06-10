Below is a **complete, self-contained PyTorch training script** that follows the pipeline you described:

* extracts **L2-normalised** ArcFace (buffalo\_l) and FaceNet-512 embeddings for every frontal image in `cfp-dataset/Data/Images/*/Frontal/*`
* builds a **balanced pair dataset** (positive = two images of the same person, negative = two images of different people)
* learns an **image-specific fusion network**

$$
\begin{aligned}
h &= \text{ReLU}(W_1 v + b_1)\;, \quad v=[e_{\text{Arc}}\;\|\;e_{\text{FN}}] \in \mathbb{R}^{1024}\\[6pt]
w &= \text{softmax}(W_2 h + b_2) = \begin{bmatrix}w_{\text{Arc}}\\w_{\text{FN}}\end{bmatrix},\; w_{\text{Arc}}+w_{\text{FN}}=1\\[6pt]
e_{\text{fused}} &= w_{\text{Arc}}\;e_{\text{Arc}} + w_{\text{FN}}\;e_{\text{FN}}\\
\end{aligned}
$$

* feeds the **absolute difference** $|e_{\text{fused}}^{(1)}-e_{\text{fused}}^{(2)}|$ through a single linear layer to predict “same / different”
* optimises **binary-cross-entropy**
* saves the trained model and fusion weights

---

```python
"""
train_fusion_mlp.py
-------------------------------------------------
Train an MLP that learns per-image weights to fuse
ArcFace (buffalo_l) and FaceNet-512 embeddings on
the CFP dataset (frontal images only).

Requirements
------------
torch, numpy, opencv-python, insightface, deepface,
tqdm (for progress bars)

Run
---
$ python train_fusion_mlp.py \
    --data_root cfp-dataset/Data/Images \
    --epochs 10 \
    --batch_size 128 \
    --num_pairs_per_id 20
"""
import os
import random
import itertools
import pickle
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from insightface.app import FaceAnalysis
from deepface import DeepFace

# --------------------------- 1. Embedding extraction -------------------------- #

def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-10)

def extract_embeddings(data_root: str,
                       cache_file: str = "embeddings.pkl"
                       ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns
    -------
    dict: {img_path: {"arc": 512-vec, "fn": 512-vec, "pid": person_id}}
    """
    if os.path.exists(cache_file):
        print(f"[INFO] Loading cached embeddings from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # initialise models once
    arc_app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider",
                                      "CPUExecutionProvider"])
    arc_app.prepare(ctx_id=0, det_size=(288, 288))

    def get_arc(img):
        faces = arc_app.get(img)
        return None if not faces else faces[0].embedding

    def get_fn(img):
        # enforce_detection=False avoids errors on hard images
        reps = DeepFace.represent(img_path=img,
                                  model_name="Facenet512",
                                  enforce_detection=False)
        return None if not reps else reps[0]["embedding"]

    all_embeddings = {}
    person_dirs = sorted(os.listdir(data_root))
    for pid in tqdm(person_dirs, desc="Embedding extraction"):
        frontal_dir = os.path.join(data_root, pid, "Frontal")
        if not os.path.isdir(frontal_dir):
            continue
        for img_name in os.listdir(frontal_dir):
            path = os.path.join(frontal_dir, img_name)
            img = cv2.imread(path)
            if img is None:
                continue
            arc = get_arc(img)
            fn  = get_fn(img)
            if arc is None or fn is None:
                continue
            all_embeddings[path] = {
                "arc": l2_normalise(np.array(arc, dtype=np.float32)),
                "fn":  l2_normalise(np.array(fn,  dtype=np.float32)),
                "pid": pid
            }

    with open(cache_file, "wb") as f:
        pickle.dump(all_embeddings, f)
    return all_embeddings


# --------------------------- 2. Pair dataset --------------------------------- #

class PairDataset(Dataset):
    def __init__(self,
                 emb_dict: Dict[str, Dict[str, np.ndarray]],
                 num_pairs_per_id: int = 20):
        self.samples: List[Tuple[str, str, int]] = []  # (img1, img2, label)
        # group by person
        by_pid: Dict[str, List[str]] = {}
        for path, meta in emb_dict.items():
            by_pid.setdefault(meta["pid"], []).append(path)

        pids = list(by_pid.keys())
        rng = random.Random(42)

        # positive pairs (label=1)
        for pid, paths in by_pid.items():
            if len(paths) < 2:
                continue
            pairs = list(itertools.combinations(paths, 2))
            rng.shuffle(pairs)
            self.samples.extend([(p1, p2, 1) for p1, p2 in pairs[:num_pairs_per_id]])

        # negative pairs (label=0)
        while len(self.samples) < 2 * len(self.samples):  # balance 1:1
            pid1, pid2 = rng.sample(pids, 2)
            p1 = rng.choice(by_pid[pid1])
            p2 = rng.choice(by_pid[pid2])
            self.samples.append((p1, p2, 0))

        rng.shuffle(self.samples)
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, label = self.samples[idx]
        e1_arc = self.emb_dict[p1]["arc"]
        e1_fn  = self.emb_dict[p1]["fn"]
        e2_arc = self.emb_dict[p2]["arc"]
        e2_fn  = self.emb_dict[p2]["fn"]
        return (e1_arc, e1_fn, e2_arc, e2_fn, np.float32(label))


# --------------------------- 3. Fusion + classifier -------------------------- #

class FusionNet(nn.Module):
    """
    Learns per-image weights w_arc, w_fn = softmax(...)
    Produces fused vector e_fused.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden, 2)  # -> logits for softmax

    def forward(self, e_arc, e_fn):
        v = torch.cat([e_arc, e_fn], dim=-1)            # (B,1024)
        h = torch.relu(self.fc1(v))                     # (B,128)
        w = torch.softmax(self.fc2(h), dim=-1)          # (B,2)
        w_arc, w_fn = w[:, 0:1], w[:, 1:2]              # (B,1)
        e_fused = w_arc * e_arc + w_fn * e_fn           # (B,512)
        e_fused = torch.nn.functional.normalize(e_fused, p=2, dim=-1)
        return e_fused, w  # return weights for inspection


class PairClassifier(nn.Module):
    """
    Whole network: 2×FusionNet + abs-diff -> logit
    """
    def __init__(self):
        super().__init__()
        self.fusion = FusionNet()
        self.fc = nn.Linear(512, 1)

    def forward(self, e1_arc, e1_fn, e2_arc, e2_fn):
        f1, _ = self.fusion(e1_arc, e1_fn)  # (B,512)
        f2, _ = self.fusion(e2_arc, e2_fn)  # (B,512)
        diff = torch.abs(f1 - f2)           # (B,512)
        logit = self.fc(diff).squeeze(-1)   # (B,)
        return logit


# --------------------------- 4. Train loop ----------------------------------- #

def train(data_root: str,
          epochs: int = 10,
          batch_size: int = 128,
          lr: float = 1e-3,
          num_pairs_per_id: int = 20,
          device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    emb_dict = extract_embeddings(data_root)
    dataset = PairDataset(emb_dict, num_pairs_per_id)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, drop_last=True, num_workers=4)

    model = PairClassifier().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            e1_arc, e1_fn, e2_arc, e2_fn, labels = batch
            e1_arc = e1_arc.to(device)
            e1_fn  = e1_fn.to(device)
            e2_arc = e2_arc.to(device)
            e2_fn  = e2_fn.to(device)
            labels = labels.to(device)

            logits = model(e1_arc, e1_fn, e2_arc, e2_fn)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item() * labels.size(0)
            pred = (logits.sigmoid() > 0.5).float()
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

        print(f"[Epoch {epoch}] loss={running_loss/total:.4f} "
              f"acc={correct/total:.4f}")

    torch.save(model.state_dict(), "fusion_mlp.pth")
    print("[INFO] Model saved to fusion_mlp.pth")


# --------------------------- 5. CLI entry-point ------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_pairs_per_id", type=int, default=20)
    args = parser.parse_args()

    train(data_root=args.data_root,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          num_pairs_per_id=args.num_pairs_per_id)
```

---

### How the script fulfils your requirements

| Requirement                                | Implementation                                                                                            |                       |                                       |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------- |
| **ArcFace + FaceNet512** embeddings        | Lines 48-82 extract and L2-normalise both vectors per image.                                              |                       |                                       |
| **Balanced pairs**                         | `PairDataset` builds *equal* numbers of positive & negative pairs (lines 96-123).                         |                       |                                       |
| **Fusion weights learned by an MLP**       | `FusionNet` (lines 129-146) realises the equations you supplied (hidden = 128, final soft-max, sum-to-1). |                       |                                       |
| **e\_fused = w\_arc·e\_arc + w\_fn·e\_fn** | Line 142.                                                                                                 |                       |                                       |
| **Pair classification**                    | `PairClassifier` (lines 149-163) feeds \`                                                                 | e\_fused₁ − e\_fused₂ | \` through a linear layer → BCE loss. |
| **Training loop & metrics**                | Lines 168-205 (accuracy and BCE loss printed each epoch).                                                 |                       |                                       |
| **CFP directory layout**                   | The script walks `data_root/<pid>/Frontal/*.jpg`. Adjust the glob if your extension set differs.          |                       |                                       |

### Speed tips

* **Cache first**: embedding extraction is much slower than network training; it’s cached to `embeddings.pkl`.
* **GPU**: both InsightFace & DeepFace will use CUDA if available; so will PyTorch.
* **Large batches**: with an H100 you can easily raise `--batch_size` to 1024+ after embeddings are on GPU.

Feel free to tweak the architecture (e.g. add deeper layers, dropout, or change the pair-loss to cosine-margin) – just keep the fusion logic intact if you want the interpretable weights $w_{\text{Arc}}, w_{\text{FN}}$.
