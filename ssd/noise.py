"""현장 소음 섞기 — 지정 SNR(dB)로 신호에 소음을 더한다. 순수 함수(시드 고정)."""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal 전력 대비 noise를 snr_db가 되도록 스케일해 더한다. noise는 길이를 맞춰 자르거나 반복한다.

    무음에 가까운 클립(rms≈0)은 그대로 둔다 — 그 SNR은 정의되지 않는다.
    """
    n = np.resize(noise, signal.shape).astype(np.float32)
    s, r = rms(signal), rms(n)
    if s <= 1e-5 or r < 1e-9:  # 완전 무음은 rms가 sqrt(1e-12)=1e-6이라 "<"로는 안 걸렸다
        return signal.copy()
    gain = s / (r * 10 ** (snr_db / 20))
    out = signal + gain * n
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.99).astype(np.float32) if peak > 1.0 else out.astype(np.float32)


def factory_noise(n: int, sr: int, seed: int) -> np.ndarray:
    """합성 공장 소음 — 브라운 잡음(저주파 우세) + 60Hz 계열 전원 험(120·240·360Hz) + 무작위 충격음."""
    rng = np.random.default_rng(seed)
    brown = np.cumsum(rng.standard_normal(n)).astype(np.float64)
    k = min(2048, n)  # 커널이 신호보다 길면 mode="same"이 커널 길이를 돌려준다(짧은 입력에서 길이가 틀어짐)
    brown -= np.convolve(brown, np.ones(k) / k, mode="same")  # 느린 표류 제거(고역 통과 근사)
    brown /= np.max(np.abs(brown)) + 1e-9
    t = np.arange(n) / sr
    f0 = 120 * (1 + 0.01 * rng.standard_normal())
    hum = sum(a * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) for k, a in ((1, 0.5), (2, 0.3), (3, 0.15)))
    clicks = np.zeros(n)
    for i in rng.integers(0, n, size=max(1, n // sr * 3)):
        L = min(int(0.01 * sr), n - i)
        clicks[i:i + L] += rng.uniform(0.5, 1.0) * np.exp(-np.arange(L) / (0.002 * sr)) * rng.choice((-1, 1))
    x = 0.6 * brown + 0.5 * hum + 0.4 * clicks
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
