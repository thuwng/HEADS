# coding=utf-8
"""
Vẽ lại visualization từ dự đoán đã lưu (không cần train/predict lại).

  PYTHONPATH=. python tools/visualize_from_run.py --run_dir logs/funsd-final-B-SegBoot-hipos0-seed42
  (tuỳ chọn: --dataset funsd|cord  --out_name visualization_v2  --max_docs 10  --no_pdf)

Yêu cầu: lần chạy đó đã lưu <run_dir>/test_raw_predictions.npz (bước 2.4 trong hướng dẫn).
"""
import argparse
import json
import os
import sys

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.getcwd())
from layoutlmft.data.visualize_predictions import visualize_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dataset", default="funsd", choices=["funsd", "cord"])
    ap.add_argument("--out_name", default="visualization")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--no_pdf", action="store_true")
    a = ap.parse_args()

    z = np.load(os.path.join(a.run_dir, "test_raw_predictions.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    if a.dataset == "funsd":
        import layoutlmft.data.funsd as mod
    else:
        import layoutlmft.data.cord as mod
    raw_test = load_dataset(os.path.abspath(mod.__file__))["test"]
    if "image" in raw_test.column_names:
        raw_test = raw_test.remove_columns(["image"])
    if meta.get("max_test_samples"):
        raw_test = raw_test.select(range(meta["max_test_samples"]))

    groups = z["pred_groups"] if "pred_groups" in z.files else None
    types = z["pred_types"] if "pred_types" in z.files else None
    res = visualize_run(
        os.path.join(a.run_dir, a.out_name), raw_test, meta["label_list"], z["pred_ids"],
        list(z["orig_word_id"]), list(z["doc_idx"]),
        text_column=meta["text_column"], label_column=meta["label_column"],
        pred_groups=groups, pred_types=types, type_names=meta.get("type_names"),
        max_docs=a.max_docs, make_pdf=not a.no_pdf,
    )
    print("Xong:", os.path.join(a.run_dir, a.out_name), res)


if __name__ == "__main__":
    main()
