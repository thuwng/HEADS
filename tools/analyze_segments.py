# coding=utf-8
"""
Chẩn đoán phân đoạn — tách lỗi do NHÓM (grouping) và lỗi do GÁN LOẠI (typing).

1) Phân tích một lần chạy SegBoot v1 đã có test_raw_predictions.npz:
     PYTHONPATH=. python tools/analyze_segments.py --run_dir logs/funsd-final-B-SegBoot-hipos0-seed42
2) Đo chất lượng phân đoạn theo dòng bằng heuristic (không học) trên train và test:
     PYTHONPATH=. python tools/analyze_segments.py --heuristic_lines

Chỉ số (gold segment = mọi item FUNSD, kể cả 'other', lấy từ cột entity_ids):
  seg_exact_P/R/F1 : segment dự đoán trùng khớp hoàn toàn segment gold
  pair_P/R/F1      : trên các cặp từ "cùng segment"
  merge_rate       : % segment dự đoán chứa từ của >= 2 segment gold   (GỘP NHẦM – có hại cho box)
  split_rate       : % segment gold bị chia ra >= 2 segment dự đoán     (TÁCH NHẦM – ít hại)
  span_recall_noO  : % entity gold (H/Q/A) được khoanh đúng tập từ, BỎ QUA loại
  typed_F1         : entity F1 có tính loại (chỉ với --run_dir)
  => span_recall_noO thấp  : nút thắt là NHÓM;  span cao nhưng typed thấp : nút thắt là GÁN LOẠI
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.getcwd())
from layoutlmft.data.order_utils import gold_entities  # noqa: E402


def comb2(n):
    return n * (n - 1) // 2


def seg_metrics(pred_lab, gold_lab):
    """pred_lab/gold_lab: list nhãn segment cho từng từ (cùng độ dài)."""
    pairs = Counter(zip(pred_lab, gold_lab))
    tp = sum(comb2(c) for c in pairs.values())
    pp = sum(comb2(c) for c in Counter(pred_lab).values())
    gp = sum(comb2(c) for c in Counter(gold_lab).values())
    pred_sets, gold_sets = defaultdict(set), defaultdict(set)
    for i, (p, g) in enumerate(zip(pred_lab, gold_lab)):
        pred_sets[p].add(i); gold_sets[g].add(i)
    P_sets = {frozenset(s) for s in pred_sets.values()}
    G_sets = {frozenset(s) for s in gold_sets.values()}
    exact = len(P_sets & G_sets)
    merge = sum(1 for s in pred_sets.values() if len({gold_lab[i] for i in s}) >= 2)
    split = sum(1 for s in gold_sets.values() if len({pred_lab[i] for i in s}) >= 2)
    return dict(tp=tp, pp=pp, gp=gp, exact=exact, n_pred=len(P_sets), n_gold=len(G_sets),
                merge=merge, split=split, P_sets=P_sets)


def summarize(acc):
    P = acc["tp"] / max(1, acc["pp"]); R = acc["tp"] / max(1, acc["gp"])
    eP = acc["exact"] / max(1, acc["n_pred"]); eR = acc["exact"] / max(1, acc["n_gold"])
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    out = {"pair_P": P, "pair_R": R, "pair_F1": f(P, R),
           "seg_exact_P": eP, "seg_exact_R": eR, "seg_exact_F1": f(eP, eR),
           "merge_rate": acc["merge"] / max(1, acc["n_pred"]),
           "split_rate": acc["split"] / max(1, acc["n_gold"]),
           "span_recall_noO": acc["span_hit"] / max(1, acc["n_ent"])}
    if acc.get("typed_n_pred"):
        tP = acc["typed_tp"] / max(1, acc["typed_n_pred"]); tR = acc["typed_tp"] / max(1, acc["n_ent"])
        out.update({"typed_P": tP, "typed_R": tR, "typed_F1": f(tP, tR),
                    "type_err_given_correct_span": acc["type_err"] / max(1, acc["span_hit"])})
    return out


def load_split(split):
    import layoutlmft.data.funsd as mod
    ds = load_dataset(os.path.abspath(mod.__file__))[split]
    return ds.remove_columns(["image"]) if "image" in ds.column_names else ds


def load_build_visual_lines():
    spec = importlib.util.spec_from_file_location("rfc", os.path.join("examples", "run_funsd_cord.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.build_visual_lines


def new_acc():
    return defaultdict(int)


def add_doc(acc, pred_lab, ex, label_names, typed_pred=None):
    gold_lab = list(ex["entity_ids"])
    m = seg_metrics(pred_lab, gold_lab)
    for k in ("tp", "pp", "gp", "exact", "n_pred", "n_gold", "merge", "split"):
        acc[k] += m[k]
    labs = [label_names[l] for l in ex["ner_tags"]]
    ents = gold_entities(labs)                               # {(type, frozenset)}
    acc["n_ent"] += len(ents)
    span_hits = {ws for _, ws in ents if ws in m["P_sets"]}
    acc["span_hit"] += len(span_hits)
    if typed_pred is not None:
        acc["typed_n_pred"] += len(typed_pred)
        acc["typed_tp"] += len(ents & typed_pred)
        gold_type = {ws: t for t, ws in ents}
        acc["type_err"] += sum(1 for t, ws in typed_pred if ws in span_hits and gold_type[ws] != t)


def analyze_run(run_dir):
    z = np.load(os.path.join(run_dir, "test_raw_predictions.npz"), allow_pickle=True)
    if "pred_groups" not in z.files:
        sys.exit("npz không có pred_groups (không phải lần chạy SegBoot v1).")
    meta = json.loads(str(z["meta"]))
    test = load_split("test")
    label_names = meta["label_list"]
    type_names = meta["type_names"]
    groups, types = z["pred_groups"], z["pred_types"]
    per_doc_lab = defaultdict(dict); per_doc_typed = defaultdict(set)
    for f, (wids, d) in enumerate(zip(z["orig_word_id"], z["doc_idx"])):
        gs = defaultdict(set); gt = {}
        for t, w in enumerate(wids):
            if w < 0 or t >= groups.shape[1] or groups[f, t] < 0:
                continue
            key = (f, int(groups[f, t]))
            per_doc_lab[int(d)][int(w)] = key
            gs[key].add(int(w)); gt[key] = int(types[f, t])
        for key, ws in gs.items():
            if gt[key] > 0:
                per_doc_typed[int(d)].add((type_names[gt[key]], frozenset(ws)))
    acc = new_acc()
    for d in range(len(test)):
        ex = test[d]
        n = len(ex["tokens"])
        lab = [per_doc_lab[d].get(i, ("miss", i)) for i in range(n)]
        add_doc(acc, lab, ex, label_names, per_doc_typed[d])
    return summarize(acc)


def analyze_heuristic(split):
    bvl = load_build_visual_lines()
    ds = load_split(split)
    label_names = ds.features["ner_tags"].feature.names
    acc = new_acc()
    for d in range(len(ds)):
        ex = ds[d]
        n = len(ex["bboxes"])
        lab = list(range(n))
        for li, line in enumerate(bvl(ex["bboxes"])):
            for i in line["indices"]:
                lab[i] = n + li
        add_doc(acc, lab, ex, label_names)
    return summarize(acc)


def show(title, r):
    print(f"\n=== {title} ===")
    for k, v in r.items():
        print(f"  {k:30s} {100 * v:6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--heuristic_lines", action="store_true")
    a = ap.parse_args()
    if a.run_dir:
        show(f"SegBoot v1 groups – {a.run_dir}", analyze_run(a.run_dir))
    if a.heuristic_lines:
        show("Heuristic visual lines – TRAIN", analyze_heuristic("train"))
        show("Heuristic visual lines – TEST", analyze_heuristic("test"))
