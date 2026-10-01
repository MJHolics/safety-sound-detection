"""조건별(깨끗함·합성 소음·실제 기계음 × SNR) 특징 캐시 — MFCC 통계·log-mel·CLAP 임베딩·CLAP 제로샷 점수.

파형은 원본 44.1kHz에서 소음을 섞은 뒤 16kHz(MFCC·CNN)와 48kHz(CLAP)로 리샘플한다. 같은 조건·같은 클립은
항상 같은 소음을 받는다(시드 = 클립 번호). 실제 기계음은 테스트 클립과 다른 폴드의 클립에서 가져온다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import MACHINE_NOISE, ROOT, Clip, classes, load, meta
from .noise import factory_noise, mix_at_snr

CACHE = ROOT / "cache"
CONDITIONS = ("clean", "synth20", "synth10", "synth0", "machine20", "machine10", "machine0")
SR_SMALL, SR_CLAP, SR_SRC = 16_000, 48_000, 44_100
CLAP_ID = "laion/clap-htsat-unfused"


def parse(cond: str) -> tuple[str, float | None]:
    if cond == "clean":
        return "clean", None
    kind = "synth" if cond.startswith("synth") else "machine"
    return kind, float(cond[len(kind):])


def _machine_pool() -> dict[int, list[Clip]]:
    """폴드별 기계음 클립 — 테스트 클립 폴드 f에는 폴드 (f mod 5)+1의 기계음을 쓴다."""
    pool: dict[int, list[Clip]] = {}
    for c in meta():
        if c.category in MACHINE_NOISE:
            pool.setdefault(c.fold, []).append(c)
    return pool


def waveform(i: int, clip: Clip, cond: str, machine_pool: dict[int, list[Clip]]) -> np.ndarray:
    """조건 cond의 44.1kHz 파형."""
    y = load(clip, SR_SRC)
    kind, snr = parse(cond)
    if kind == "clean":
        return y
    if kind == "synth":
        n = factory_noise(len(y), SR_SRC, seed=10_000 + i)
    else:
        donors = machine_pool[clip.fold % 5 + 1]
        n = load(donors[i % len(donors)], SR_SRC)
    return mix_at_snr(y, n, snr)


def mfcc_stats(y16: np.ndarray) -> np.ndarray:
    """MFCC 40 + 델타 40의 평균·표준편차 = 160차원(고전 기준선)."""
    import librosa

    m = librosa.feature.mfcc(y=y16, sr=SR_SMALL, n_mfcc=40)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def logmel(y16: np.ndarray) -> np.ndarray:
    """64 mel × 약 157 프레임(hop 512) log-mel, float16 저장."""
    import librosa

    S = librosa.feature.melspectrogram(y=y16, sr=SR_SMALL, n_fft=1024, hop_length=512, n_mels=64)
    return librosa.power_to_db(S, ref=1.0, top_db=None).astype(np.float16)


def _proj(out):
    """transformers 5.x의 get_*_features는 출력 객체를 돌려준다 — 투영 임베딩은 pooler_output(forward의 *_embeds와 일치 확인)."""
    return out if hasattr(out, "norm") else out.pooler_output


class Clap:
    """CLAP(LAION, HTSAT) — 오디오 임베딩과 텍스트 프롬프트 제로샷."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(CLAP_ID).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(CLAP_ID)
        prompts = [f"This is a sound of {c.replace('_', ' ')}." for c in classes()]
        t = self.proc(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            te = _proj(self.model.get_text_features(**t))
        self.text = torch.nn.functional.normalize(te, dim=-1)

    def embed(self, batch48: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(정규화 오디오 임베딩, 제로샷 코사인 점수[클래스 50])."""
        torch = self.torch
        x = self.proc(audio=batch48, sampling_rate=SR_CLAP, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ae = torch.nn.functional.normalize(_proj(self.model.get_audio_features(**x)), dim=-1)
        return ae.float().cpu().numpy(), (ae @ self.text.T).float().cpu().numpy()


def build(cond: str, clap: Clap | None, batch: int = 32) -> dict[str, np.ndarray]:
    """조건 cond의 특징을 계산하거나 캐시에서 읽는다."""
    import librosa

    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{cond}.npz"
    if f.exists():
        return dict(np.load(f))
    pool = _machine_pool()
    mf, lm, emb, zs = [], [], [], []
    buf: list[np.ndarray] = []

    def flush():
        if buf and clap is not None:
            e, z = clap.embed(buf)
            emb.append(e)
            zs.append(z)
        buf.clear()

    for i, c in enumerate(meta()):
        y = waveform(i, c, cond, pool)
        y16 = librosa.resample(y, orig_sr=SR_SRC, target_sr=SR_SMALL)
        mf.append(mfcc_stats(y16))
        lm.append(logmel(y16))
        buf.append(librosa.resample(y, orig_sr=SR_SRC, target_sr=SR_CLAP))
        if len(buf) == batch:
            flush()
    flush()
    out = {"mfcc": np.stack(mf), "logmel": np.stack(lm)}
    if clap is not None:
        out["clap_emb"] = np.concatenate(emb)
        out["clap_zs"] = np.concatenate(zs)
    np.savez(f, **out)
    return out


def labels() -> tuple[np.ndarray, np.ndarray]:
    m = meta()
    return np.array([c.target for c in m]), np.array([c.fold for c in m])


def cache_path() -> Path:
    return CACHE
