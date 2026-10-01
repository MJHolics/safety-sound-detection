"""ESC-50 로딩 — 2,000클립·50종·공식 5-fold. 오디오는 data/ESC-50-master(라이선스 CC BY-NC, 저장소 미포함)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ESC = ROOT / "data" / "ESC-50-master"

# 산업안전 경보 대상 — 모델을 돌리기 전에 고정했다(2026-10-02).
SAFETY = ("glass_breaking", "siren", "crackling_fire", "clock_alarm", "car_horn")
# 실제 기계 배경음 — 평가 때만 섞는다(학습 증강에는 합성 소음만 쓴다).
MACHINE_NOISE = ("engine", "vacuum_cleaner", "washing_machine")


@dataclass(frozen=True)
class Clip:
    filename: str
    fold: int
    target: int
    category: str


@lru_cache(maxsize=1)
def meta() -> tuple[Clip, ...]:
    with open(ESC / "meta" / "esc50.csv", encoding="utf-8") as f:
        return tuple(Clip(r["filename"], int(r["fold"]), int(r["target"]), r["category"]) for r in csv.DictReader(f))


def classes() -> list[str]:
    names = {c.target: c.category for c in meta()}
    return [names[i] for i in range(len(names))]


def load(clip: Clip, sr: int) -> np.ndarray:
    """모노 float32, sr로 리샘플(원본 44.1kHz, 5초)."""
    import librosa
    import soundfile as sf

    y, src = sf.read(ESC / "audio" / clip.filename, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=src, target_sr=sr) if src != sr else y
