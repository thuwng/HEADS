# coding=utf-8
"""
layoutlmft/data/segboot_utils.py — tiện ích cho SegBoot (dữ liệu, giao thức, metric).
Không phụ thuộc mô hình; có thể chạy test độc lập.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object


# =============================================================================
# 1. Cột cho từng cửa sổ (gọi trong tokenize_and_align_labels)
# =============================================================================

def segboot_token_columns(word_ids: Sequence[Optional[int]], word_eids: Sequence[int]):
    """
    word_ids : tokenized_inputs.word_ids(batch_index)  (chỉ số từ trong danh sách ĐÃ reorder)
    word_eids: entity id GOLD theo từng từ ĐÃ reorder (FUNSD item id; gồm cả item 'other')
    Trả về:
      token_node_pos[t] = vị trí sub-token đầu (trong cửa sổ) của từ chứa token t, -1 nếu special
      word_entity_id[t] = eid tại sub-token đầu của từ, -1 nơi khác
    Quy ước "sub-token đầu" trùng với align_word_level_columns (w != prev), kể cả khi
    cửa sổ tràn bắt đầu giữa một từ.
    """
    token_node_pos, word_entity_id = [], []
    prev, cur_first = None, -1
    for t, w in enumerate(word_ids):
        if w is None:
            token_node_pos.append(-1)
            word_entity_id.append(-1)
            prev = None
            continue
        if w != prev:
            cur_first = t
            word_entity_id.append(int(word_eids[w]))
        else:
            word_entity_id.append(-1)
        token_node_pos.append(cur_first)
        prev = w
    return token_node_pos, word_entity_id


def line_level_boxes(bboxes, build_lines_fn):
    """Segment box KHÔNG học: mỗi từ nhận union box của dòng thị giác chứa nó
    (build_visual_lines của run_funsd_cord.py; chỉ dùng hình học, không nhìn nhãn).
    Dùng cho baseline B0-L (--bbox_level line)."""
    out = [list(b) for b in bboxes]
    if len(bboxes) == 0:
        return out
    for line in build_lines_fn(bboxes):
        idx = list(line["indices"])
        if not idx:
            continue
        ub = [min(bboxes[i][0] for i in idx), min(bboxes[i][1] for i in idx),
              max(bboxes[i][2] for i in idx), max(bboxes[i][3] for i in idx)]
        for i in idx:
            out[i] = ub
    return out


def perturb_order(order: List[int], rng: random.Random, prob: float = 0.5, window: int = 6,
                  frac: float = 0.15) -> List[int]:
    """Augmentation thứ tự đọc (chỉ dùng cho TRAIN): xáo trộn cục bộ một phần các đoạn ngắn,
    mô phỏng lỗi chen dòng của OCR/XY-cut. Nhãn được retag lại sau đó bằng retag_bio_after_reorder."""
    if prob <= 0 or rng.random() > prob or len(order) < 3:
        return list(order)
    out = list(order)
    n_ops = max(1, int(frac * len(out) / max(window, 1)))
    for _ in range(n_ops):
        s = rng.randrange(0, max(1, len(out) - window))
        seg = out[s:s + window]
        rng.shuffle(seg)
        out[s:s + window] = seg
    return out


# =============================================================================
# 2. Giao thức: tách dev từ train (FUNSD không có dev chính thức)
# =============================================================================

def make_dev_split(ds_dict, dev_ratio: float, seed: int, fold: int = -1, num_folds: int = 5):
    """
    dev_ratio > 0 và fold < 0  : tách ngẫu nhiên dev_ratio của train làm 'validation'.
    fold >= 0                   : k-fold (num_folds), fold thứ `fold` làm 'validation'.
    Tập 'test' giữ nguyên, KHÔNG bao giờ dùng để chọn siêu tham số/checkpoint.
    """
    from datasets import DatasetDict
    train = ds_dict["train"]
    n = len(train)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if fold >= 0:
        folds = [idx[i::num_folds] for i in range(num_folds)]
        dev_idx = sorted(folds[fold])
    else:
        dev_idx = sorted(idx[: max(1, int(round(dev_ratio * n)))])
    dev_set = set(dev_idx)
    tr_idx = [i for i in range(n) if i not in dev_set]
    out = DatasetDict({k: v for k, v in ds_dict.items()})
    out["train"] = train.select(tr_idx)
    out["validation"] = train.select(dev_idx)
    return out


class SegBootEpsCallback(TrainerCallback):
    """Scheduled sampling: eps giảm tuyến tính từ eps_start -> eps_end trong decay_frac * max_steps."""

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.0, decay_frac: float = 0.6):
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac

    def _set(self, model, step, max_steps):
        span = max(1.0, self.decay_frac * max_steps)
        r = min(1.0, step / span)
        eps = self.eps_start + (self.eps_end - self.eps_start) * r
        core = getattr(getattr(model, "module", model), "segboot", None)
        if core is not None:
            core.ss_eps = float(eps)

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._set(model, 0, state.max_steps)

    def on_step_begin(self, args, state, control, model=None, **kw):
        self._set(model, state.global_step, state.max_steps)

# =============================================================================
# 3. Dự đoán & metric
# =============================================================================

def unpack_predictions(predictions):
    """Trainer trả tuple (logits, pred_groups, pred_types) với SegBoot; ndarray với mô hình khác."""
    if isinstance(predictions, (tuple, list)):
        logits = predictions[0]
        groups = predictions[1] if len(predictions) > 1 else None
        types = predictions[2] if len(predictions) > 2 else None
        return logits, groups, types
    return predictions, None, None
