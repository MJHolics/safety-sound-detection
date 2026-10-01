"""소음 섞기·파워 영역 가산·McNemar — 순수 단위테스트(데이터셋·GPU 불필요)."""
import numpy as np
import pytest

from ssd.noise import factory_noise, mix_at_snr, rms


@pytest.mark.parametrize("snr", [20.0, 10.0, 0.0, -5.0])
def test_mix_hits_target_snr_without_clipping(snr):
    s = (0.01 * np.sin(np.linspace(0, 2000, 80_000))).astype(np.float32)  # 작은 진폭 — 정규화가 안 걸리게
    n = factory_noise(80_000, 16_000, seed=1) * 0.01
    out = mix_at_snr(s, n, snr)
    assert np.max(np.abs(out)) <= 1.0
    assert 20 * np.log10(rms(s) / rms(out - s)) == pytest.approx(snr, abs=0.05)


def test_mix_never_exceeds_full_scale():
    s = np.ones(16_000, np.float32) * 0.9
    out = mix_at_snr(s, factory_noise(16_000, 16_000, seed=2), 0.0)
    assert np.max(np.abs(out)) <= 0.99 + 1e-6


def test_silent_clip_is_left_alone():
    s = np.zeros(1_000, np.float32)
    assert np.array_equal(mix_at_snr(s, factory_noise(1_000, 16_000, 3), 10.0), s)


def test_factory_noise_is_deterministic_per_seed():
    a, b, c = factory_noise(8_000, 16_000, 7), factory_noise(8_000, 16_000, 7), factory_noise(8_000, 16_000, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_power_domain_noise_hits_snr():
    torch = pytest.importorskip("torch")
    from ssd.cnn import _add_noise_db

    sig_db = torch.full((1, 64, 100), 0.0)      # 파워 1
    noise_db = torch.full((1, 64, 100), -20.0)  # 파워 0.01 — 스케일은 함수가 맞춘다
    out = _add_noise_db(sig_db, noise_db, torch.tensor([10.0]))
    added = 10 ** (out / 10) - 1.0               # 더해진 소음 파워
    assert float(10 * torch.log10(1.0 / added.mean())) == pytest.approx(10.0, abs=1e-3)


def test_mcnemar():
    from run_bench import mcnemar_p

    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    only_a, only_b, p = mcnemar_p(a, b)
    assert (only_a, only_b) == (10, 0) and p == pytest.approx(2 / 2 ** 10)
