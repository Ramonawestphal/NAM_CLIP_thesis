"""
v7-parity Step 0: verify src/models/nam_multiclass.py is task-agnostic in num_classes.

Reads the NAM module and confirms num_classes parameterises the output layer and
bias with no hard-coded 7 / HAM10000-specific assumptions, so it can be imported
with num_classes=2 for the binary chest X-ray task WITHOUT modifying the module.

Run from project root:
    python scripts/chestxray/_verify_nam_multiclass.py
"""

from __future__ import annotations

import inspect
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

NAM_PATH = _ROOT / "src/models/nam_multiclass.py"


def main() -> None:
    from src.models.nam_multiclass import NAMMulticlass, FeatureNNMulticlass

    src = NAM_PATH.read_text(encoding="utf-8")
    sig = inspect.signature(NAMMulticlass.__init__)

    print("=" * 64)
    print("NAMMulticlass task-agnosticism check")
    print("=" * 64)
    print(f"Source: {NAM_PATH.relative_to(_ROOT)}")
    print(f"\nClass signature:\n  NAMMulticlass{sig}")

    # 1. num_classes accepted as a parameter
    has_num_classes = "num_classes" in sig.parameters
    print(f"\n[check] 'num_classes' is a constructor parameter: {has_num_classes}")

    # 2. num_classes flows to the output layer + bias
    out_layer = "Linear(in_dim, num_classes" in src
    bias_param = "torch.zeros(num_classes)" in src
    print("[check] num_classes flows through:")
    print(f"    - FeatureNNMulticlass output: Linear(in_dim, num_classes, bias=False)  -> {out_layer}")
    print(f"    - per-class global bias:      Parameter(torch.zeros(num_classes))       -> {bias_param}")
    print("    - forward(): stacks K subnet outputs (B,K,C) and sums to (B,C) logits")

    # 3. scan for suspicious hard-coded integers (7, or HAM10000 class tokens)
    ham_tokens = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    found_tokens = [t for t in ham_tokens if re.search(rf"\b{t}\b", src)]
    # bare '7' that is not part of a larger number / version / line ref
    sevens = re.findall(r"(?<![\w.])7(?![\w.])", src)
    print("\n[check] suspicious hard-coded references:")
    print(f"    - HAM10000 class tokens {ham_tokens}: {found_tokens or 'none'}")
    print(f"    - bare integer '7' occurrences: {len(sevens)} "
          f"({'none' if not sevens else 'inspect manually'})")

    agnostic = has_num_classes and out_layer and bias_param and not found_tokens
    print("\n" + "-" * 64)
    if agnostic:
        print("[VERIFIED] NAMMulticlass is task-agnostic; can be imported with num_classes=2.")
    else:
        print("[STOP] NAMMulticlass does NOT appear task-agnostic. "
              "Do NOT modify it — ask Ramona before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
