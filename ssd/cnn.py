"""소형 log-mel CNN — 처음부터 학습, 엣지용(파라미터 약 10만).

증강: 시간 이동 + 시간·주파수 마스크(SpecAugment). noise_aug=True면 학습 중 합성 공장 소음을 log-mel **파워 영역**에서
무작위 SNR(0~20dB)로 더한다(파워 가산 근사). 실제 기계음은 학습에 쓰지 않는다 — 평가 때 처음 보는 소음 유형.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 50):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.f = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):  # x: (B, 1, 64, T) — 정규화된 log-mel
        h = self.f(x)
        return self.head(h.mean(dim=(2, 3)))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _norm(x: torch.Tensor, mu: float, sd: float) -> torch.Tensor:
    return (x - mu) / sd


def _add_noise_db(lm: torch.Tensor, noise_db: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """log-mel(dB) 신호에 소음 log-mel(dB)을 파워 영역에서 지정 SNR로 더한다."""
    ps = 10 ** (lm / 10)
    pn = 10 ** (noise_db / 10)
    g = ps.mean(dim=(1, 2), keepdim=True) / (pn.mean(dim=(1, 2), keepdim=True) * 10 ** (snr_db.view(-1, 1, 1) / 10))
    return 10 * torch.log10(ps + g * pn + 1e-10)


def train(lm_train: np.ndarray, y_train: np.ndarray, noise_bank: np.ndarray | None, seed: int,
          epochs: int = 60, device: str = "cuda") -> tuple[SmallCNN, float, float]:
    """(모델, 정규화 평균, 표준편차). lm_train: (N, 64, T) dB."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.tensor(lm_train.astype(np.float32), device=device)
    Y = torch.tensor(y_train, device=device)
    NB = torch.tensor(noise_bank.astype(np.float32), device=device) if noise_bank is not None else None
    mu, sd = float(X.mean()), float(X.std())
    m = SmallCNN().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(X) + 63) // 64))
    T = X.shape[-1]
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(X), device=device)
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]
            x = X[idx]
            if NB is not None:  # 절반에만 소음 — 깨끗한 소리 성능을 잃지 않게
                k = idx.numel() // 2
                nb = NB[torch.randint(len(NB), (k,), device=device)]
                snr = torch.tensor(rng.uniform(0, 20, k), device=device, dtype=torch.float32)
                x = torch.cat([_add_noise_db(x[:k], nb, snr), x[k:]])
            shift = int(rng.integers(0, T))
            x = torch.roll(x, shifts=shift, dims=-1)
            x = _norm(x, mu, sd).unsqueeze(1)
            for _m in range(2):  # SpecAugment: 시간 마스크 2, 주파수 마스크 2
                t0 = int(rng.integers(0, T - 20)); x[..., t0:t0 + int(rng.integers(0, 20))] = 0
                f0 = int(rng.integers(0, 56)); x[:, :, f0:f0 + int(rng.integers(0, 8)), :] = 0
            loss = nn.functional.cross_entropy(m(x), Y[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return m.eval(), mu, sd


@torch.no_grad()
def predict(m: SmallCNN, lm: np.ndarray, mu: float, sd: float, device: str = "cuda") -> np.ndarray:
    X = _norm(torch.tensor(lm.astype(np.float32), device=device), mu, sd).unsqueeze(1)
    return torch.cat([m(X[i:i + 256]) for i in range(0, len(X), 256)]).softmax(-1).cpu().numpy()
