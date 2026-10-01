"""엣지 비용 — 5초 클립 1개를 CPU 1스레드로 판정하는 데 드는 시간과 모델 크기.

  python edge_bench.py   # → results/edge.json

전처리(MFCC·log-mel·CLAP 특징 추출) 시간까지 포함한 종단 지연을 클립 50개로 잰다(처음 3개는 예열로 제외).
CNN은 PyTorch와 ONNX Runtime(FP32·동적 INT8) 둘 다 잰다. 정확도는 run_bench.py가 재므로 여기선 같은 클립에서
ONNX INT8이 PyTorch와 같은 클래스를 내는지만 확인한다(일치율).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ssd.data import ROOT, meta  # noqa: E402
from ssd.features import CLAP_ID, SR_CLAP, SR_SMALL, logmel, mfcc_stats  # noqa: E402

OUT = ROOT / "results"


def timeit(fn, xs, warm=3):
    for x in xs[:warm]:
        fn(x)
    ts = []
    for x in xs[warm:]:
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.median(ts)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def main() -> int:
    import librosa
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sklearn.linear_model import LogisticRegression

    from ssd.cnn import SmallCNN, n_params

    torch.set_num_threads(1)
    clips = [c for c in meta() if c.fold == 1][:53]
    from ssd.data import load

    w16 = [load(c, SR_SMALL) for c in clips]
    w48 = [load(c, SR_CLAP) for c in clips]
    res = {}

    # MFCC + LR (가중치는 무작위 학습으로 대체 — 지연만 잰다)
    X = np.stack([mfcc_stats(w) for w in w16])
    lr = LogisticRegression(max_iter=2000).fit(np.vstack([X] * 2), np.arange(len(X) * 2) % 50)
    res["mfcc_lr"] = {**timeit(lambda w: lr.predict_proba(mfcc_stats(w)[None]), w16),
                      "params": int(lr.coef_.size + lr.intercept_.size)}

    # 소형 CNN — PyTorch CPU / ONNX FP32 / ONNX INT8
    cache = ROOT / "cache"
    ck = torch.load(cache / "cnn_aug_fold1.pt", weights_only=False)  # run_bench.py가 남긴 fold 1 학습 모델
    m = SmallCNN()
    m.load_state_dict(ck["state"])
    m.eval()
    mu, sd = ck["mu"], ck["sd"]

    class Normed(torch.nn.Module):  # 정규화까지 그래프에 넣어 ONNX 입력을 log-mel(dB) 그대로 받게 한다
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net((x - mu) / sd)

    m = Normed(m).eval()
    onnx32, onnx8 = cache / "cnn_fp32.onnx", cache / "cnn_int8.onnx"
    dummy = torch.zeros(1, 1, 64, 157)
    torch.onnx.export(m, dummy, str(onnx32), input_names=["x"], output_names=["y"], dynamo=False)
    # onnxruntime 양자화는 shape inference 중간 파일을 다시 여는데, 한글 경로(AI개발)에서 실패한다 → ASCII 임시 폴더에서 수행
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        a32, a8 = Path(td) / "m32.onnx", Path(td) / "m8.onnx"
        shutil.copy(onnx32, a32)
        quantize_dynamic(str(a32), str(a8), weight_type=QuantType.QInt8)
        shutil.copy(a8, onnx8)
        # 정적 양자화(QDQ) — 학습 폴드(2~5) log-mel 200개로 활성값 범위를 보정한다. 테스트 폴드 1은 보정에 안 쓴다.
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

        from ssd.features import build as _build, labels as _labels

        _y, _f = _labels()
        calib = _build("clean", None)["logmel"][_f != 1][:200].astype(np.float32)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter({"x": x[None, None]} for x in calib)

            def get_next(self):
                return next(self.it, None)

        a8s = Path(td) / "m8s.onnx"
        quantize_static(str(a32), str(a8s), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
        shutil.copy(a8s, cache / "cnn_int8_static.onnx")

    def cnn_torch(w):
        with torch.no_grad():
            return m(torch.tensor(logmel(w).astype(np.float32))[None, None])

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s32 = ort.InferenceSession(str(onnx32), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(onnx8), so, providers=["CPUExecutionProvider"])
    run32 = lambda w: s32.run(None, {"x": logmel(w).astype(np.float32)[None, None]})  # noqa: E731
    run8 = lambda w: s8.run(None, {"x": logmel(w).astype(np.float32)[None, None]})  # noqa: E731
    res["cnn_torch"] = {**timeit(cnn_torch, w16), "params": n_params(m.net)}
    res["cnn_onnx_fp32"] = {**timeit(run32, w16), "bytes": onnx32.stat().st_size}
    res["cnn_onnx_int8"] = {**timeit(run8, w16), "bytes": onnx8.stat().st_size}
    # fold 1 테스트 400클립(깨끗한 소리) — FP32 vs INT8 정확도
    from ssd.features import build, labels

    y, folds = labels()
    lm1 = build("clean", None)["logmel"][folds == 1].astype(np.float32)
    y1 = y[folds == 1]
    p32 = np.array([s32.run(None, {"x": x[None, None]})[0].argmax() for x in lm1])
    p8 = np.array([s8.run(None, {"x": x[None, None]})[0].argmax() for x in lm1])
    res["cnn_onnx_fp32"]["fold1_acc"] = round(float((p32 == y1).mean()), 4)
    res["cnn_onnx_int8"]["fold1_acc"] = round(float((p8 == y1).mean()), 4)
    res["cnn_onnx_int8"]["argmax_agree_with_fp32"] = round(float((p32 == p8).mean()), 4)
    s8s = ort.InferenceSession(str(cache / "cnn_int8_static.onnx"), so, providers=["CPUExecutionProvider"])
    run8s = lambda w: s8s.run(None, {"x": logmel(w).astype(np.float32)[None, None]})  # noqa: E731
    p8s = np.array([s8s.run(None, {"x": x[None, None]})[0].argmax() for x in lm1])
    res["cnn_onnx_int8_static"] = {**timeit(run8s, w16), "bytes": (cache / "cnn_int8_static.onnx").stat().st_size,
                                   "fold1_acc": round(float((p8s == y1).mean()), 4),
                                   "argmax_agree_with_fp32": round(float((p32 == p8s).mean()), 4)}
    res["logmel_only"] = timeit(logmel, w16)

    # CLAP 오디오 인코더 (CPU)
    from transformers import ClapModel, ClapProcessor

    cm = ClapModel.from_pretrained(CLAP_ID).eval()
    cp = ClapProcessor.from_pretrained(CLAP_ID)

    def clap(w):
        with torch.no_grad():
            return cm.get_audio_features(**cp(audio=[w], sampling_rate=SR_CLAP, return_tensors="pt"))

    audio_params = sum(p.numel() for n, p in cm.named_parameters() if n.startswith(("audio_model", "audio_projection")))
    res["clap_audio_cpu"] = {**timeit(clap, w48), "params": int(audio_params)}
    res["note"] = "CPU 1스레드, 5초 클립, 전처리 포함 종단 지연, 클립 50개(예열 3 제외)"

    OUT.mkdir(exist_ok=True)
    (OUT / "edge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
