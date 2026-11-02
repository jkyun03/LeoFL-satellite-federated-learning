# CIFAR-10 이미지를 사전학습 ResNet18으로 512차원 임베딩해 .npz로 캐시하고, MLP 학습용 텐서 데이터셋으로 제공한다.
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.models as models


class EmbedTabularDataset(Dataset):
    """
    Simulator와 호환되는 tabular-like dataset
    - X: [N, D] float32
    - Y: [N] long
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_classes: int,
        class_weight: torch.Tensor = None,
    ):
        self.X = X.contiguous()
        self.Y = Y.contiguous()
        self.n_features = int(self.X.shape[1])
        self.n_classes = int(n_classes)
        self.class_weight = class_weight

        self.features = None
        self.scaler = None
        self.category_encoder = None

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _compute_balanced_class_weight(Y: torch.Tensor, n_classes: int) -> torch.Tensor:
    y = Y.detach().cpu().numpy().astype(int)
    counts = np.bincount(y, minlength=n_classes)
    N = counts.sum()

    w = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    w[nonzero] = float(N) / (float(n_classes) * counts[nonzero].astype(np.float32))
    return torch.from_numpy(w).float()


@torch.no_grad()
def build_cifar10_embeddings(
    root: str = "LeoFL/CIFAR10/data",
    cache_path: str = "LeoFL/CIFAR10/cache/cifar10_resnet18_512.npz",
    batch_size: int = 256,
    num_workers: int = 4,
    device: torch.device = None,
):
    """
    CIFAR-10 -> ResNet18 pretrained -> 512-d embedding
    train/test를 합쳐 하나의 full dataset으로 만든 뒤 반환
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(root, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 캐시 로드
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        X = torch.from_numpy(arr["X"]).float()
        Y = torch.from_numpy(arr["Y"]).long()
        n_classes = int(arr["n_classes"])
        cw = _compute_balanced_class_weight(Y, n_classes)
        return EmbedTabularDataset(X, Y, n_classes=n_classes, class_weight=cw)

    # 2) transform
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds = CIFAR10(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3) pretrained ResNet18 -> fc 제거
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    feats = []
    labels = []

    for loader in [train_loader, test_loader]:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = model(x)  # [B, 512]
            feats.append(f.cpu())
            labels.append(y.cpu())

    X = torch.cat(feats, dim=0).float()
    Y = torch.cat(labels, dim=0).long()
    n_classes = 10

    np.savez(
        cache_path,
        X=X.numpy(),
        Y=Y.numpy(),
        n_classes=np.int32(n_classes),
    )

    cw = _compute_balanced_class_weight(Y, n_classes)
    return EmbedTabularDataset(X, Y, n_classes=n_classes, class_weight=cw)
