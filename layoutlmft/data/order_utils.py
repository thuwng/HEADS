# coding=utf-8
"""
layoutlmft/data/order_utils.py

(1) retag_bio_after_reorder: sau khi XY-Cut đảo thứ tự từ, nhãn BIO gốc không
    còn hợp lệ (I- có thể đứng trước B-). Hàm này gán lại B/I theo thứ tự MỚI,
    dựa trên entity gốc. Chỉ xử lý TARGET, không phải input -> không leakage.

(2) entity_set_prf: đánh giá entity-level KHÔNG phụ thuộc thứ tự đọc.
    Gold entity = (type, tập chỉ số từ gốc) lấy từ annotation (thứ tự gốc).
    Pred entity = span BIO trên chuỗi đã reorder, ánh xạ về chỉ số từ gốc.
    Exact match trên (type, tập từ). Cùng một thước đo cho mọi setting,
    nên số ở setting oracle và setting XY-Cut so sánh được với nhau.

Chỉ hỗ trợ BIO (FUNSD, CORD). Với BIOES cần mở rộng _split.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _split(label: str):
    if label == "O":
        return None, None
    prefix, etype = label.split("-", 1)
    if prefix not in ("B", "I"):
        raise ValueError(f"Only BIO is supported, got label {label!r}")
    return prefix, etype


def entity_ids_from_bio(word_labels: Sequence[str]) -> List[int]:
    """Entity id cho từng từ theo thứ tự GỐC; -1 cho O."""
    ent, eid, prev_type = [], -1, None
    for lab in word_labels:
        prefix, etype = _split(lab)
        if etype is None:
            ent.append(-1)
            prev_type = None
            continue
        if prefix == "B" or etype != prev_type:
            eid += 1
        ent.append(eid)
        prev_type = etype
    return ent


def retag_bio_after_reorder(word_labels: Sequence[str], order: Sequence[int]) -> List[str]:
    """
    word_labels: nhãn chuỗi theo thứ tự GỐC.
    order: order[k] = chỉ số gốc của từ đứng ở vị trí k sau reorder.
    Trả về nhãn BIO hợp lệ theo thứ tự mới. Entity bị XY-Cut tách rời sẽ có
    nhiều B- (mô hình bị phạt đúng mức ở bước đánh giá entity_set_prf).
    """
    ent = entity_ids_from_bio(word_labels)
    out, prev = [], None
    for j in order:
        _, etype = _split(word_labels[j])
        if etype is None:
            out.append("O")
            prev = None
            continue
        out.append(("I-" if ent[j] == prev else "B-") + etype)
        prev = ent[j]
    return out


def gold_entities(word_labels: Sequence[str]):
    ent = entity_ids_from_bio(word_labels)
    groups, types = defaultdict(set), {}
    for i, (e, lab) in enumerate(zip(ent, word_labels)):
        if e < 0:
            continue
        groups[e].add(i)
        types[e] = _split(lab)[1]
    return {(types[e], frozenset(ws)) for e, ws in groups.items()}


def entities_from_sequence(seq: Sequence[Tuple[int, str]]):
    """seq: [(orig_word_id, predicted_label), ...] theo thứ tự đọc của mô hình."""
    ents, cur, cur_t = [], None, None
    for wid, lab in seq:
        prefix, etype = _split(lab)
        if etype is None:
            if cur:
                ents.append((cur_t, frozenset(cur)))
            cur, cur_t = None, None
            continue
        if cur is None or prefix == "B" or etype != cur_t:
            if cur:
                ents.append((cur_t, frozenset(cur)))
            cur, cur_t = [wid], etype
        else:
            cur.append(wid)
    if cur:
        ents.append((cur_t, frozenset(cur)))
    return set(ents)


def entity_set_prf(
    predictions,
    label_list: Sequence[str],
    orig_word_id: Sequence[Sequence[int]],
    doc_idx: Sequence[int],
    gold_word_labels: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """
    predictions: (N, L, C) logits hoặc (N, L) ids từ trainer.predict.
    orig_word_id: theo từng feature (window); >=0 tại sub-token đầu của từ.
    doc_idx: chỉ số tài liệu (toàn cục, từ map(with_indices=True)) của từng feature.
    gold_word_labels: nhãn chuỗi theo thứ tự GỐC, cùng thứ tự với dataset test.
    Các window của cùng tài liệu phải liền nhau và đúng thứ tự (mặc định của .map).
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3:
        preds = preds.argmax(-1)

    per_doc = defaultdict(list)
    for f, (wids, d) in enumerate(zip(orig_word_id, doc_idx)):
        for t, w in enumerate(wids):
            if w >= 0 and t < preds.shape[1]:
                per_doc[int(d)].append((int(w), label_list[int(preds[f, t])]))

    tp = n_pred = n_gold = 0
    per_type = defaultdict(lambda: [0, 0, 0])  # tp, pred, gold
    for d, labels in enumerate(gold_word_labels):
        gold = gold_entities(labels)
        pred = entities_from_sequence(per_doc.get(d, []))
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
    out = {"entity_precision": P, "entity_recall": R, "entity_f1": F}
    for t, (a, p, g) in sorted(per_type.items()):
        out[f"entity_f1_{t}"] = prf(a, p, g)[2]
    return out


if __name__ == "__main__":
    # Sanity tests
    labs = ["B-QUESTION", "I-QUESTION", "B-ANSWER", "O", "B-HEADER", "I-HEADER"]
    assert retag_bio_after_reorder(labs, list(range(6))) == labs
    # đảo 2 từ trong QUESTION -> vẫn 1 entity, B đứng đầu theo thứ tự mới
    assert retag_bio_after_reorder(labs, [1, 0, 2, 3, 4, 5])[:2] == ["B-QUESTION", "I-QUESTION"]
    # HEADER bị tách bởi ANSWER -> 2 B-
    assert retag_bio_after_reorder(labs, [4, 2, 5, 0, 1, 3]) == \
        ["B-HEADER", "B-ANSWER", "B-HEADER", "B-QUESTION", "I-QUESTION", "O"]
    g = gold_entities(labs)
    p = entities_from_sequence([(1, "B-QUESTION"), (0, "I-QUESTION"), (2, "B-ANSWER"),
                                (3, "O"), (4, "B-HEADER"), (5, "I-HEADER")])
    assert g == p, (g, p)
    print("order_utils sanity tests passed")