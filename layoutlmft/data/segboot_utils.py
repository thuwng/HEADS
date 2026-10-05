# coding=utf-8
"""
layoutlmft/data/segboot_utils.py — tiện ích cho SegBoot (dữ liệu, giao thức, metric).
Không phụ thuộc mô hình; có thể chạy test độc lập.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from layoutlmft.data.order_utils import gold_entities

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


def type_prior_from_features(labels_rows, word_start_rows, label_list: Sequence[str]) -> List[float]:
    """Tần suất loại (O + các entity type theo thứ tự sorted) ở mức TỪ trên tập train."""
    types = ["O"] + sorted({l[2:] for l in label_list if l != "O"})
    cnt = np.ones(len(types))  # +1 smoothing
    for labs, ws in zip(labels_rows, word_start_rows):
        for l, s in zip(labs, ws):
            if s == 1 and l != -100:
                name = label_list[l]
                cnt[0 if name == "O" else types.index(name[2:])] += 1
    return (cnt / cnt.sum()).tolist()


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


def entity_set_prf_groups(pred_groups, pred_types, type_names: Sequence[str],
                          orig_word_id, doc_idx, gold_word_labels) -> Dict[str, float]:
    """
    Entity-level P/R/F1 trên TẬP từ gốc (exact match type + tập từ), không qua BIO.
    Thực thể bị cắt bởi cửa sổ 512 token được tính như hai thực thể (giống seqeval của LayoutLMv3).
    """
    pred_groups = np.asarray(pred_groups)
    pred_types = np.asarray(pred_types)
    per_doc_pred = defaultdict(set)
    for f, (wids, d) in enumerate(zip(orig_word_id, doc_idx)):
        groups = defaultdict(set)
        gtype = {}
        for t, w in enumerate(wids):
            if w < 0 or t >= pred_groups.shape[1]:
                continue
            g, ty = int(pred_groups[f, t]), int(pred_types[f, t])
            if g < 0 or ty <= 0:          # type 0 = O
                continue
            groups[g].add(int(w))
            gtype[g] = type_names[ty]
        for g, ws in groups.items():
            per_doc_pred[int(d)].add((gtype[g], frozenset(ws)))

    tp = n_pred = n_gold = 0
    per_type = defaultdict(lambda: [0, 0, 0])
    for d, labels in enumerate(gold_word_labels):
        gold = gold_entities(labels)
        pred = per_doc_pred.get(d, set())
        hit = gold & pred
        tp += len(hit); n_pred += len(pred); n_gold += len(gold)
        for t, _ in hit:
            per_type[t][0] += 1
        for t, _ in pred:
            per_type[t][1] += 1
        for t, _ in gold:
            per_type[t][2] += 1

    def prf(a, p, g):
        pr = a / p if p else 0.0
        rc = a / g if g else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    P, R, F = prf(tp, n_pred, n_gold)
    out = {"group_entity_precision": P, "group_entity_recall": R, "group_entity_f1": F}
    for t, (a, p, g) in sorted(per_type.items()):
        out[f"group_entity_f1_{t}"] = prf(a, p, g)[2]
    return out
