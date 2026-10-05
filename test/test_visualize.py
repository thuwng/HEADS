# coding=utf-8
"""PYTHONPATH=. python tests/test_visualize.py  -> sinh ảnh mẫu ở /tmp/vis_test và kiểm tra F1 khớp metric."""
import os
import numpy as np
from PIL import Image, ImageDraw
from layoutlmft.data.visualize_predictions import visualize_run
from layoutlmft.data.order_utils import entity_set_prf
from layoutlmft.data.segboot_utils import entity_set_prf_groups

LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
OUT = "/tmp/vis_test"


def make_doc():
    os.makedirs(OUT, exist_ok=True)
    W, H = 762, 1000
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    lines = [("HEADER", ["ACME", "TOBACCO", "REPORT"], 60), ("QUESTION", ["Name:"], 160),
             ("ANSWER", ["John", "Smith"], 160), ("QUESTION", ["Date:"], 230), ("ANSWER", ["12/09/1999"], 230),
             ("QUESTION", ["Brand", "code:"], 300), ("ANSWER", ["RJ", "100"], 300), ("O", ["page", "1", "of", "2"], 900)]
    xpos = {"HEADER": 250, "QUESTION": 80, "ANSWER": 330, "O": 300}
    tokens, bboxes, tags = [], [], []
    for ty, ws, y in lines:
        x = xpos[ty]
        for i, w in enumerate(ws):
            bw = 14 * len(w) + 10
            d.text((x + 2, y + 4), w, fill="black")
            bboxes.append([int(1000 * x / W), int(1000 * y / H), int(1000 * (x + bw) / W), int(1000 * (y + 22) / H)])
            tokens.append(w)
            tags.append(0 if ty == "O" else LABELS.index(("B-" if i == 0 else "I-") + ty))
            x += bw + 8
    path = os.path.join(OUT, "doc0.png"); img.save(path)
    return [{"tokens": tokens, "bboxes": bboxes, "ner_tags": tags, "image_path": path}], [LABELS[t] for t in tags]


def main():
    raw, gold = make_doc()
    n = len(gold)
    pred = list(gold)
    pred[3] = "B-HEADER"                        # sai loại
    pred[5] = "B-ANSWER"                        # tách entity -> 2 lỗi span
    pred[7] = "O"                               # bỏ sót
    pred[14] = "B-QUESTION"; pred[15] = "I-QUESTION"   # FP trên vùng other
    ids = np.array([[0] + [LABELS.index(p) for p in pred] + [0]])
    owid = [[-1] + list(range(n)) + [-1]]
    r = visualize_run(os.path.join(OUT, "bio"), raw, LABELS, ids, owid, [0])
    e = entity_set_prf(ids, LABELS, owid, [0], [gold])
    assert abs(r["vis_f1"] - e["entity_f1"]) < 1e-9, (r, e)

    types = ["O", "ANSWER", "HEADER", "QUESTION"]
    eids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7]
    grp = np.full((1, n + 2), -1); ty = np.full((1, n + 2), -1)
    for i in range(n):
        grp[0, i + 1] = eids[i]
        ty[0, i + 1] = 0 if gold[i] == "O" else types.index(gold[i][2:])
    ty[0, 8] = 0; grp[0, 10] = 99
    r2 = visualize_run(os.path.join(OUT, "segboot"), raw, LABELS, ids, owid, [0],
                       pred_groups=grp, pred_types=ty, type_names=types)
    e2 = entity_set_prf_groups(grp, ty, types, owid, [0], [gold])
    assert abs(r2["vis_f1"] - e2["group_entity_f1"]) < 1e-9, (r2, e2)
    print(f"ok  BIO F1 {r['vis_f1']:.4f} == metric | SegBoot F1 {r2['vis_f1']:.4f} == metric")
    print("ảnh mẫu:", sorted(os.listdir(os.path.join(OUT, "bio"))))


if __name__ == "__main__":
    main()
