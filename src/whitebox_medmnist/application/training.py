"""Training and evaluation as in MedMNIST v2: Adam, cross-entropy, ACC and AUC on the test split."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def scores(probs: np.ndarray, labels: np.ndarray) -> dict:
    acc = float((probs.argmax(1) == labels).mean())
    auc = float(roc_auc_score(labels, probs[:, 1]) if probs.shape[1] == 2 else
                roc_auc_score(labels, probs, multi_class="ovr"))
    return {"acc": acc, "auc": auc}


def fit(model, train, val, epochs, lr, device, milestones=(), log=print):
    """Keep the epoch with the best validation AUC (MedMNIST's model selection)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(milestones), gamma=0.1)
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        val_scores = scores(*predict(model, val, device))
        if val_scores["auc"] > best:
            best, best_state = val_scores["auc"], {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"  epoch {epoch}: val acc {val_scores['acc']:.3f} auc {val_scores['auc']:.3f}")
    model.load_state_dict(best_state)
    return model
