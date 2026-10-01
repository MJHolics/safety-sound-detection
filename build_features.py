"""조건 7종 특징 캐시 생성 — python build_features.py (cache/<조건>.npz)."""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ssd.features import CONDITIONS, Clap, build  # noqa: E402

clap = Clap()
for c in CONDITIONS:
    t = time.time()
    out = build(c, clap)
    print(c, {k: v.shape for k, v in out.items()}, f"{time.time() - t:.0f}s", flush=True)
print("BUILD_DONE")
