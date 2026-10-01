"""위험음 인식 벤치 — 공식 5-fold, 모델 7종 × 조건 7종.

  python build_features.py   # 먼저 특징 캐시
  python run_bench.py        # → results/bench.json, results/preds.npz

모델:
  mfcc_lr / mfcc_lr_aug   MFCC 통계 + 로지스틱 (aug: 학습 폴드의 합성 소음 특징을 더해 학습)
  cnn / cnn_aug           소형 log-mel CNN 처음부터 학습 (aug: 합성 공장 소음 파워 가산)
  clap_zs                 CLAP 제로샷 — "This is a sound of {class}." 학습 없음
  clap_lr / clap_lr_aug   CLAP 오디오 임베딩 + 로지스틱
학습에는 깨끗한 소리와 합성 공장 소음만 쓴다. 실제 기계음(machine*)은 평가에만 — 처음 보는 소음 유형.
위험음 = SAFETY 5종(ssd/data.py). 검출 = 예측 클래스가 SAFETY에 속함.
"""
from __future__ import annotations

import json
import sys
import time
from math import comb

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ssd.data import ROOT, SAFETY, classes  # noqa: E402
from ssd.features import CONDITIONS, SR_SMALL, build, labels, logmel  # noqa: E402
from ssd.noise import factory_noise  # noqa: E402

OUT = ROOT / "results"
MODELS = ("mfcc_lr", "mfcc_lr_aug", "cnn", "cnn_aug", "clap_zs", "clap_lr", "clap_lr_aug")


def logreg(Xtr, ytr, Xte_list):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xtr, ytr)
    return [clf.predict_proba(X) for X in Xte_list]


def mcnemar_p(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int, float]:
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def main() -> int:
    feats = {c: build(c, None) for c in CONDITIONS}  # 캐시에서 읽기만(없으면 build_features.py 먼저)
    y, folds = labels()
    names = classes()
    safe_ids = np.array([names.index(s) for s in SAFETY])
    noise_bank = np.stack([logmel(factory_noise(5 * SR_SMALL, SR_SMALL, seed=90_000 + k)) for k in range(200)])

    preds = {m: {c: np.zeros((len(y), len(names)), np.float32) for c in CONDITIONS} for m in MODELS}
    t_model: dict[str, float] = {m: 0.0 for m in MODELS}
    for f in range(1, 6):
        tr, te = folds != f, folds == f
        t0 = time.time()
        for kind, key in (("mfcc", "mfcc"), ("clap", "clap_emb")):
            Xtr = feats["clean"][key][tr]
            outs = logreg(Xtr, y[tr], [feats[c][key][te] for c in CONDITIONS])
            for c, p in zip(CONDITIONS, outs):
                preds[f"{kind}_lr"][c][te] = p
            Xaug = np.concatenate([Xtr] + [feats[c][key][tr] for c in ("synth20", "synth10", "synth0")])
            outs = logreg(Xaug, np.tile(y[tr], 4), [feats[c][key][te] for c in CONDITIONS])
            for c, p in zip(CONDITIONS, outs):
                preds[f"{kind}_lr_aug"][c][te] = p
        t_model["mfcc_lr"] += time.time() - t0
        from ssd.cnn import predict, train

        for name, bank in (("cnn", None), ("cnn_aug", noise_bank)):
            t0 = time.time()
            m, mu, sd = train(feats["clean"]["logmel"][tr], y[tr], bank, seed=f)
            for c in CONDITIONS:
                preds[name][c][te] = predict(m, feats[c]["logmel"][te], mu, sd)
            t_model[name] += time.time() - t0
            if f == 1 and name == "cnn_aug":  # 엣지 벤치(ONNX INT8 정확도)용으로 fold 1 모델을 남긴다
                import torch

                torch.save({"state": m.state_dict(), "mu": mu, "sd": sd}, ROOT / "cache" / "cnn_aug_fold1.pt")
        for c in CONDITIONS:
            z = feats[c]["clap_zs"][te]
            preds["clap_zs"][c][te] = np.exp(z * 100) / np.exp(z * 100).sum(1, keepdims=True)
        print(f"fold {f} done", {k: round(v) for k, v in t_model.items() if v}, flush=True)

    summary: dict = {"n": int(len(y)), "safety_classes": list(SAFETY), "conditions": list(CONDITIONS), "models": {}}
    is_safe = np.isin(y, safe_ids)
    for m in MODELS:
        summary["models"][m] = {}
        for c in CONDITIONS:
            yhat = preds[m][c].argmax(1)
            ok = yhat == y
            fold_acc = [float(ok[folds == f].mean()) for f in range(1, 6)]
            flagged = np.isin(yhat, safe_ids)
            summary["models"][m][c] = {
                "acc": round(float(ok.mean()), 4),
                "acc_fold_sd": round(float(np.std(fold_acc)), 4),
                "safety_recall": round(float(flagged[is_safe].mean()), 4),
                "safety_false_alarm": round(float(flagged[~is_safe].mean()), 4),
                "safety_missed": int((~flagged[is_safe]).sum()),
                "per_safety_class_recall": {s: round(float(flagged[y == names.index(s)].mean()), 3) for s in SAFETY},
            }
    pairs = [("mfcc_lr", "cnn"), ("cnn", "cnn_aug"), ("cnn", "clap_lr"), ("clap_zs", "clap_lr"),
             ("clap_lr", "clap_lr_aug"), ("mfcc_lr", "mfcc_lr_aug")]
    summary["mcnemar"] = {}
    for a, b in pairs:
        for c in ("clean", "machine10", "machine0"):
            oka, okb = preds[a][c].argmax(1) == y, preds[b][c].argmax(1) == y
            only_a, only_b, p = mcnemar_p(oka, okb)
            summary["mcnemar"][f"{a}|{b}|{c}"] = {"only_a": only_a, "only_b": only_b, "p": p}
    summary["train_seconds"] = {k: round(v, 1) for k, v in t_model.items()}

    OUT.mkdir(exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "preds.npz", **{f"{m}__{c}": preds[m][c] for m in MODELS for c in CONDITIONS})
    hdr = "model".ljust(12) + "".join(c.rjust(10) for c in CONDITIONS)
    print("\n정확도(50종)\n" + hdr)
    for m in MODELS:
        print(m.ljust(12) + "".join(f"{summary['models'][m][c]['acc']:10.3f}" for c in CONDITIONS))
    print("\n위험음 재현율 / 오경보\n" + hdr)
    for m in MODELS:
        print(m.ljust(12) + "".join(
            f"{summary['models'][m][c]['safety_recall']:5.2f}/{summary['models'][m][c]['safety_false_alarm']:.2f}" for c in CONDITIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
