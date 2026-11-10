"""Train a small CNN on FER2013 folders (data/train, data/test). Needs: pip install torch torchvision. GPU recommended (~15-25 min on a free Colab/Kaggle GPU).
NOTE: written for you to run; it was NOT executed in the authoring sandbox (no PyTorch there). Expect roughly 63-68% test accuracy; a fine-tuned ResNet18 reaches ~68-72%."""
import torch, torch.nn as nn, torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

class EmotionCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25))
        self.f = nn.Sequential(blk(1, 64), blk(64, 128), blk(128, 256))        # 48 -> 24 -> 12 -> 6
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(256 * 36, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n))
    def forward(self, x): return self.c(self.f(x))

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm = T.Normalize([0.5], [0.5])
    tr_t = T.Compose([T.Grayscale(), T.RandomHorizontalFlip(), T.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)), T.ToTensor(), norm])
    te_t = T.Compose([T.Grayscale(), T.ToTensor(), norm])
    full = ImageFolder('data/train', tr_t)                                       # class order is alphabetical = CLASSES in assistant.py
    n_val = int(0.1 * len(full)); tr, va = random_split(full, [len(full) - n_val, n_val], torch.Generator().manual_seed(0))
    va.dataset = ImageFolder('data/train', te_t)                                 # note: va shares indices but uses no augmentation
    te = ImageFolder('data/test', te_t)
    L = lambda d, s: DataLoader(d, 128, shuffle=s, num_workers=2)
    counts = torch.bincount(torch.tensor(full.targets)).float()
    w = (counts.sum() / (len(counts) * counts)).to(dev)                          # class weights (fixes tiny 'disgust' class)
    m = EmotionCNN().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    E = 40; sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=E * len(L(tr, True)))
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1); best = 0
    def evaluate(dl):
        m.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in dl: P += m(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
        return P, Y
    for e in range(E):
        m.train()
        for x, y in L(tr, True):
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); lossf(m(x), y).backward(); opt.step(); sch.step()
        P, Y = evaluate(L(va, False)); acc = sum(p == y for p, y in zip(P, Y)) / len(Y); print(f'epoch {e+1}/{E} val acc {acc:.3f}')
        if acc > best: best = acc; torch.save(m.state_dict(), 'emotion_cnn.pt')
    m.load_state_dict(torch.load('emotion_cnn.pt')); P, Y = evaluate(L(te, False))
    print(classification_report(Y, P, target_names=full.classes, digits=3))