# coding=utf-8
"""
layoutlmft/data/visualize_predictions.py

Trực quan hoá kết quả dự đoán trên tập test: mỗi tài liệu sinh MỘT ảnh gồm hai panel.

  ┌──────────── GOLD ────────────┬──────────── PREDICTION ────────────┐
  │ entity gold, tô theo loại     │ entity dự đoán, tô theo loại DỰ ĐOÁN│
  │                               │ viền XANH LÁ  = đúng hoàn toàn      │
  │                               │ viền ĐỎ       = sai (nhãn ghi lý do)│
  │                               │ viền CAM đứt  = entity gold bị bỏ sót│
  │                               │ gạch chân đỏ  = từ bị gán sai loại   │
  └───────────────────────────────┴────────────────────────────────────┘
  Thanh trên: tên file, TP/FP/FN, P/R/F1 của tài liệu. Thanh dưới: chú giải.

Phân loại một entity dự đoán (loại t, tập từ S):
  OK    : (t, S) trùng khớp một entity gold                 -> True positive
  TYPE  : S trùng khớp tập từ của một entity gold nhưng sai loại
  SPAN  : S giao một phần với entity gold (thiếu/thừa từ, gộp/tách nhầm)
  FP    : S không giao entity gold nào (dự đoán trên vùng 'other')
Một entity gold không được dự đoán đúng hoàn toàn -> MISS (vẽ ở panel Prediction).

Logic ghép entity GIỐNG HỆT metric:
  * mô hình BIO   -> order_utils.entities_from_sequence   (== entity_set_prf)
  * SegBoot       -> nhóm pred_groups / pred_types         (== entity_set_prf_groups)
nên tổng TP/FP/FN trong summary.csv tái tạo đúng entity_f1 / group_entity_f1 đã báo cáo.
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from layoutlmft.data.order_utils import entities_from_sequence, gold_entities

Entity = Tuple[str, frozenset]

# ---------------------------------------------------------------------------
# Màu
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "HEADER": (241, 196, 15),     # vàng
    "QUESTION": (52, 152, 219),   # xanh dương
    "ANSWER": (155, 89, 182),     # tím
}
_EXTRA = [(26, 188, 156), (230, 126, 34), (52, 73, 94), (231, 76, 60), (149, 165, 166),
          (211, 84, 0), (39, 174, 96), (41, 128, 185), (142, 68, 173), (243, 156, 18)]
C_OK = (39, 174, 96)
C_ERR = (231, 76, 60)
C_MISS = (230, 126, 34)
C_TEXT = (20, 20, 20)


def type_color(t: str) -> Tuple[int, int, int]:
    if t in TYPE_COLORS:
        return TYPE_COLORS[t]
    return _EXTRA[sum(map(ord, t)) % len(_EXTRA)]


def short(t: str) -> str:
    return {"HEADER": "H", "QUESTION": "Q", "ANSWER": "A"}.get(t, t[:10])


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:  # Pillow cũ
        return draw.textsize(text, font=font)


# ---------------------------------------------------------------------------
# 1. Thu thập entity dự đoán theo từng tài liệu (giống metric)
# ---------------------------------------------------------------------------

def collect_predictions(pred_ids, label_list, orig_word_id, doc_idx,
                        pred_groups=None, pred_types=None, type_names=None):
    """
    Trả về:
      doc_ents[d]  : set[(type, frozenset(orig_word_ids))]
      doc_wtype[d] : {orig_word_id: loại dự đoán ('O' nếu không thuộc entity)}
    """
    doc_ents: Dict[int, Set[Entity]] = defaultdict(set)
    doc_wtype: Dict[int, Dict[int, str]] = defaultdict(dict)
    pred_ids = np.asarray(pred_ids)
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(-1)

    if pred_groups is not None:
        pred_groups, pred_types = np.asarray(pred_groups), np.asarray(pred_types)
        for f, (wids, d) in enumerate(zip(orig_word_id, doc_idx)):
            groups, gtype = defaultdict(set), {}
            for t, w in enumerate(wids):
                if w < 0 or t >= pred_groups.shape[1]:
                    continue
                g, ty = int(pred_groups[f, t]), int(pred_types[f, t])
                name = type_names[ty] if ty > 0 else "O"
                doc_wtype[int(d)][int(w)] = name
                if g < 0 or ty <= 0:
                    continue
                groups[g].add(int(w))
                gtype[g] = name
            for g, ws in groups.items():
                doc_ents[int(d)].add((gtype[g], frozenset(ws)))
        return doc_ents, doc_wtype

    per_doc = defaultdict(list)
    for f, (wids, d) in enumerate(zip(orig_word_id, doc_idx)):
        for t, w in enumerate(wids):
            if w >= 0 and t < pred_ids.shape[1]:
                lab = label_list[int(pred_ids[f, t])]
                per_doc[int(d)].append((int(w), lab))
                doc_wtype[int(d)][int(w)] = "O" if lab == "O" else lab.split("-", 1)[1]
    for d, seq in per_doc.items():
        doc_ents[d] = entities_from_sequence(seq)
    return doc_ents, doc_wtype


# ---------------------------------------------------------------------------
# 2. Đối chiếu gold / pred
# ---------------------------------------------------------------------------

def match_entities(gold: Set[Entity], pred: Set[Entity]):
    gold_by_set = {ws: t for t, ws in gold}
    pred_status = {}
    for p in pred:
        t, ws = p
        if p in gold:
            pred_status[p] = ("OK", None)
        elif ws in gold_by_set:
            pred_status[p] = ("TYPE", gold_by_set[ws])
        else:
            overlap = [g for g in gold if g[1] & ws]
            if overlap:
                best = max(overlap, key=lambda g: len(g[1] & ws))
                pred_status[p] = ("SPAN", best[0])
            else:
                pred_status[p] = ("FP", None)
    missed = [g for g in gold if g not in pred]
    counts = {k: sum(1 for s, _ in pred_status.values() if s == k) for k in ("OK", "TYPE", "SPAN", "FP")}
    tp, n_pred, n_gold = counts["OK"], len(pred), len(gold)
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    stats = dict(n_gold=n_gold, n_pred=n_pred, tp=tp, wrong_type=counts["TYPE"],
                 wrong_span=counts["SPAN"], fp_other=counts["FP"], missed=len(missed),
                 precision=P, recall=R, f1=F)
    return pred_status, missed, stats


# ---------------------------------------------------------------------------
# 3. Vẽ
# ---------------------------------------------------------------------------

def _px_boxes(bboxes_norm, W, H):
    return [(b[0] * W / 1000.0, b[1] * H / 1000.0, b[2] * W / 1000.0, b[3] * H / 1000.0) for b in bboxes_norm]


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _dashed_rect(draw, box, color, width=2, dash=8, gap=5):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        n = int(length // (dash + gap)) + 1
        for i in range(n):
            s = i * (dash + gap) / length
            e = min(1.0, (i * (dash + gap) + dash) / length)
            draw.line([(ax + (bx - ax) * s, ay + (by - ay) * s), (ax + (bx - ax) * e, ay + (by - ay) * e)],
                      fill=color, width=width)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _label(draw, box, text, fg, bg, font, placed, prefer="above"):
    """Đặt nhãn cạnh box, tránh chồng lên các nhãn đã đặt (placed: list rect)."""
    x0, y0, x1, y1 = box
    tw, th = _text_size(draw, text, font)
    w, h = tw + 6, th + 4
    above = [(x0, y0 - h - 1 - k * (h + 1)) for k in range(4)]
    below = [(x0, y1 + 1 + k * (h + 1)) for k in range(4)]
    cands = (above + below) if prefer == "above" else (below + above)
    cands += [(x0 + w + 2, c[1]) for c in cands[:2]]
    W, H = draw.im.size
    choice = None
    for (cx, cy) in cands:
        cx = min(max(0, cx), max(0, W - w)); cy = min(max(0, cy), max(0, H - h))
        r = (cx, cy, cx + w, cy + h)
        if not any(_overlap(r, q) for q in placed):
            choice = r
            break
    if choice is None:
        cx = min(max(0, x0), max(0, W - w)); cy = max(0, y0 - h - 1)
        choice = (cx, cy, cx + w, cy + h)
    placed.append(choice)
    draw.rectangle(list(choice), fill=bg)
    draw.text((choice[0] + 3, choice[1] + 1), text, fill=fg, font=font)


def _panel(base: Image.Image, wboxes, ents_draw, outline_fn, label_fn, font, alpha=90,
           underline_words=(), missed=()):
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for (t, ws) in ents_draw:
        c = type_color(t)
        for w in ws:
            if w < len(wboxes):
                od.rectangle(wboxes[w], fill=c + (alpha,))
    img = Image.alpha_composite(img, overlay)
    d = ImageDraw.Draw(img)
    placed = []
    for e in ents_draw:
        ws = [w for w in e[1] if w < len(wboxes)]
        if not ws:
            continue
        ub = _union([wboxes[w] for w in ws])
        ub = (ub[0] - 2, ub[1] - 2, ub[2] + 2, ub[3] + 2)
        color, width = outline_fn(e)
        d.rectangle(ub, outline=color, width=width)
        txt = label_fn(e)
        if txt:
            _label(d, ub, txt, (255, 255, 255), color, font, placed, prefer="above")
    for w in underline_words:
        if w < len(wboxes):
            x0, y0, x1, y1 = wboxes[w]
            d.line([(x0, y1 + 1), (x1, y1 + 1)], fill=C_ERR, width=3)
    for (t, ws) in missed:
        ws = [w for w in ws if w < len(wboxes)]
        if not ws:
            continue
        ub = _union([wboxes[w] for w in ws])
        ub = (ub[0] - 4, ub[1] - 4, ub[2] + 4, ub[3] + 4)
        _dashed_rect(d, ub, C_MISS, width=3)
        _label(d, ub, f"MISS {short(t)}", (255, 255, 255), C_MISS, font, placed, prefer="below")
    return img.convert("RGB")


def render_document(image_path, bboxes_norm, gold: Set[Entity], pred: Set[Entity],
                    wtype_pred: Dict[int, str], wtype_gold: Dict[int, str],
                    title: str, type_list: Sequence[str]):
    base = Image.open(image_path).convert("RGB")
    W, H = base.size
    scale = 1.0 if max(W, H) >= 1000 else 1000.0 / max(W, H)   # phóng to ảnh nhỏ cho dễ đọc
    if scale != 1.0:
        base = base.resize((int(W * scale), int(H * scale)), Image.BICUBIC)
        W, H = base.size
    wboxes = _px_boxes(bboxes_norm, W, H)
    font = _font(max(11, int(H / 75)))
    big = _font(max(14, int(H / 50)))

    pred_status, missed, st = match_entities(gold, pred)

    gold_panel = _panel(base, wboxes, sorted(gold, key=lambda e: min(e[1])),
                        outline_fn=lambda e: (type_color(e[0]), 2),
                        label_fn=lambda e: short(e[0]), font=font)

    def p_outline(e):
        return (C_OK, 3) if pred_status[e][0] == "OK" else (C_ERR, 3)

    def p_label(e):
        s, g = pred_status[e]
        if s == "OK":
            return f"{short(e[0])} OK"
        if s == "TYPE":
            return f"{short(e[0])} X type(gold {short(g)})"
        if s == "SPAN":
            return f"{short(e[0])} X span(gold {short(g)})"
        return f"{short(e[0])} X FP"

    wrong_words = [w for w, gt in wtype_gold.items()
                   if wtype_pred.get(w, "O") != gt and (gt != "O" or wtype_pred.get(w, "O") != "O")]
    pred_panel = _panel(base, wboxes, sorted(pred, key=lambda e: min(e[1])),
                        outline_fn=p_outline, label_fn=p_label, font=font,
                        underline_words=wrong_words, missed=missed)

    pad, top, bottom = 16, int(H * 0.06) + 20, int(H * 0.07) + 30
    canvas = Image.new("RGB", (W * 2 + pad * 3, H + top + bottom), (255, 255, 255))
    canvas.paste(gold_panel, (pad, top))
    canvas.paste(pred_panel, (W + pad * 2, top))
    d = ImageDraw.Draw(canvas)
    head = (f"{title}   |   gold={st['n_gold']}  pred={st['n_pred']}  TP={st['tp']}  "
            f"type-err={st['wrong_type']}  span-err={st['wrong_span']}  FP={st['fp_other']}  "
            f"miss={st['missed']}   |   P={100*st['precision']:.1f}  R={100*st['recall']:.1f}  "
            f"F1={100*st['f1']:.1f}")
    d.text((pad, 8), head, fill=C_TEXT, font=big)
    d.text((pad, top - _text_size(d, "G", big)[1] - 6), "GOLD", fill=C_TEXT, font=big)
    d.text((W + pad * 2, top - _text_size(d, "G", big)[1] - 6), "PREDICTION", fill=C_TEXT, font=big)

    # chú giải
    y = top + H + 12
    x = pad
    for t in type_list:
        d.rectangle([x, y, x + 22, y + 22], fill=type_color(t))
        d.text((x + 28, y + 2), t, fill=C_TEXT, font=font)
        x += 40 + _text_size(d, t, font)[0]
    items = [(C_OK, "viền xanh: đúng"), (C_ERR, "viền đỏ: sai (type/span/FP)"),
             (C_MISS, "viền cam đứt: gold bị bỏ sót"), (C_ERR, "gạch chân đỏ: từ sai loại")]
    for c, txt in items:
        d.rectangle([x, y, x + 22, y + 22], outline=c, width=3)
        d.text((x + 28, y + 2), txt, fill=C_TEXT, font=font)
        x += 40 + _text_size(d, txt, font)[0]
    return canvas, st


# ---------------------------------------------------------------------------
# 4. Chạy cho cả tập test
# ---------------------------------------------------------------------------

def visualize_run(out_dir: str, raw_test, label_list, pred_ids, orig_word_id, doc_idx,
                  text_column: str = "tokens", label_column: str = "ner_tags", label_to_id=None,
                  pred_groups=None, pred_types=None, type_names=None,
                  max_docs: Optional[int] = None, make_pdf: bool = True, logger=None):
    os.makedirs(out_dir, exist_ok=True)
    doc_ents, doc_wtype = collect_predictions(pred_ids, label_list, orig_word_id, doc_idx,
                                              pred_groups, pred_types, type_names)
    label_to_id = label_to_id or {i: i for i in range(len(label_list))}
    type_list = sorted({l[2:] for l in label_list if l != "O"})

    rows, pages = [], []
    tot = defaultdict(int)
    n = len(raw_test) if max_docs is None else min(max_docs, len(raw_test))
    for d in range(len(raw_test)):
        ex = raw_test[d]
        gold_lab = [label_list[label_to_id[l]] for l in ex[label_column]]
        gold = gold_entities(gold_lab)
        pred = doc_ents.get(d, set())
        wtype_gold = {i: ("O" if l == "O" else l.split("-", 1)[1]) for i, l in enumerate(gold_lab)}
        name = os.path.splitext(os.path.basename(ex["image_path"]))[0]
        _, _, st = match_entities(gold, pred)
        for k in ("n_gold", "n_pred", "tp"):
            tot[k] += st[k]
        fname = ""
        if d < n:
            img, st = render_document(ex["image_path"], ex["bboxes"], gold, pred,
                                      doc_wtype.get(d, {}), wtype_gold,
                                      title=f"[{d:03d}] {name}", type_list=type_list)
            fname = f"{d:03d}_{name}_F1-{100*st['f1']:.0f}.png"
            img.save(os.path.join(out_dir, fname))
            if make_pdf:
                pages.append(img)
        rows.append(dict(doc_idx=d, name=name, file=fname, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                for k, v in st.items()}))

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["f1"]):      # tài liệu tệ nhất lên đầu
            w.writerow(r)
    if make_pdf and pages:
        pages[0].save(os.path.join(out_dir, "all_documents.pdf"), save_all=True,
                      append_images=pages[1:], resolution=100.0)

    P = tot["tp"] / tot["n_pred"] if tot["n_pred"] else 0.0
    R = tot["tp"] / tot["n_gold"] if tot["n_gold"] else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    if logger is not None:
        logger.info(f"*** Visualization: {n} ảnh tại {out_dir} | micro F1 tái tạo = {F:.6f} ***")
    return {"vis_precision": P, "vis_recall": R, "vis_f1": F}
