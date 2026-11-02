# CIFAR-10 원본 이미지(임베딩 없이)를 CNN 학습용 Dataset으로 감싸는 로더.
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10RawDataset(Dataset):
    """
    시뮬레이터와 호환되는 raw image dataset
    - X: [N, 3, 32, 32]
    - Y: [N]
    - n_classes, class_weight 제공
    - n_features는 CNN 경로에서는 사용 안 하므로 None
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
        self.n_classes = int(n_classes)
        self.class_weight = class_weight
        self.n_features = None

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


def build_cifar10_raw_datasets(
    root: str = "LeoFL/CIFAR10/data",
):
    """
    CIFAR-10 raw image dataset 반환
    - central_train: train split
    - test_dataset: test split
    """
    os.makedirs(root, exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    train_ds = CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds = CIFAR10(root=root, train=False, download=True, transform=tfm)

    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))], dim=0)
    Y_train = torch.tensor(
        [train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long
    )

    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    Y_test = torch.tensor(
        [test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long
    )

    n_classes = 10
    train_cw = _compute_balanced_class_weight(Y_train, n_classes)

    central_train = CIFAR10RawDataset(
        X=X_train,
        Y=Y_train,
        n_classes=n_classes,
        class_weight=train_cw,
    )

    test_dataset = CIFAR10RawDataset(
        X=X_test,
        Y=Y_test,
        n_classes=n_classes,
        class_weight=train_cw.clone(),
    )

    return central_train, test_dataset
