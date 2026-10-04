# coding=utf-8
"""
Chạy:  PYTHONPATH=. python tests/test_segboot.py
Kiểm tra lõi SegBoot với backbone giả (không cần trọng số LayoutLMv3).
"""
import random

import numpy as np
import torch
import torch.nn as nn

from layoutlmft.models.layoutlmv3.segboot_core import (
    SegBootCore, transitive_closure, component_boxes, component_ids, knn_pair_mask,
)
from layoutlmft.data.segboot_utils import (
    segboot_token_columns, entity_set_prf_groups, unpack_predictions, perturb_order,
    SegBootEpsCallback, type_prior_from_features,
)
from layoutlmft.data.order_utils import retag_bio_after_reorder, entity_set_prf

LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
ID2LABEL = dict(enumerate(LABELS))
H = 32


class FakeBackbone(nn.Module):
    """h = emb(input_ids) + MLP(bbox) ; ảnh = 1 CLS + 7x7 patch học được."""
    def __init__(self, vocab=200):
        super().__init__()
        self.emb = nn.Embedding(vocab, H)
        self.box = nn.Sequential(nn.Linear(4, H), nn.GELU(), nn.Linear(H, H))
        self.mix = nn.TransformerEncoder(nn.TransformerEncoderLayer(H, 4, 64, 0.0, batch_first=True), 1)
        self.img = nn.Parameter(torch.randn(1, 50, H) * 0.1)

    def forward(self, input_ids, bbox):
        x = self.emb(input_ids) + self.box(bbox.float() / 1000.0)
        x = self.mix(x)
        return x, self.img.expand(input_ids.shape[0], -1, -1)


def synthetic_doc(n_lines=6, words_per_line=4, seed=0):
    """Mỗi dòng là một entity; loại ngẫu nhiên. Trả về danh sách từ (box, eid, label)."""
    rng = random.Random(seed)
    words = []
    eid = 0
    for li in range(n_lines):
        ty = rng.choice(["HEADER", "QUESTION", "ANSWER", "O"])
        x = 50
        for wi in range(words_per_line):
            box = [x, 60 + li * 70, x + 60, 60 + li * 70 + 25]
            lab = "O" if ty == "O" else (("B-" if wi == 0 else "I-") + ty)
            words.append((box, eid, lab))
            x += 75
        eid += 1
    return words


def build_batch(words, order=None, sub_tokens=2, vocab=200, seed=0):
    """Mô phỏng tokenize: mỗi từ -> sub_tokens token; CLS/SEP ở hai đầu."""
    order = order if order is not None else list(range(len(words)))
    rng = random.Random(seed)
    rw = [words[j] for j in order]
    word_ids, input_ids, bbox = [None], [0], [[0, 0, 0, 0]]
    for wi, (box, eid, lab) in enumerate(rw):
        for s in range(sub_tokens):
            word_ids.append(wi); input_ids.append(3 + rng.randrange(vocab - 3)); bbox.append(box)
    word_ids.append(None); input_ids.append(2); bbox.append([0, 0, 0, 0])
    eids = [e for _, e, _ in rw]
    labs_orig = [l for _, _, l in words]
    labs = retag_bio_after_reorder(labs_orig, order)
    tnp, weid = segboot_token_columns(word_ids, eids)
    word_start, labels, orig_wid = [], [], []
    prev = None
    for t, w in enumerate(word_ids):
        if w is None:
            word_start.append(0); labels.append(-100); orig_wid.append(-1)
        elif w != prev:
            word_start.append(1); labels.append(LABELS.index(labs[w])); orig_wid.append(order[w])
        else:
            word_start.append(0); labels.append(-100); orig_wid.append(-1)
        prev = w
    T = lambda x: torch.tensor([x])
    return dict(input_ids=T(input_ids), bbox=T(bbox), attention_mask=T([1] * len(input_ids)),
                word_start=T(word_start), token_node_pos=T(tnp), word_entity_id=T(weid),
                labels=T(labels)), orig_wid, labs_orig


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = FakeBackbone()
        self.segboot = SegBootCore(H, ID2LABEL, k_nn=8)
        self.segboot.reset_parameters()

    def forward(self, input_ids, bbox, attention_mask, word_start, token_node_pos, labels=None,
                word_entity_id=None):
        run = lambda bb: self.bb(input_ids, bb)
        return self.segboot(run, input_ids, bbox, attention_mask, word_start, token_node_pos,
                            labels=labels, word_entity_id=word_entity_id, special_ids=(0, 2, 1))


# ---------------------------------------------------------------------------
def test_token_columns():
    word_ids = [None, 0, 0, 1, 2, 2, 2, None]
    tnp, weid = segboot_token_columns(word_ids, [7, 7, 9])
    assert tnp == [-1, 1, 1, 3, 4, 4, 4, -1], tnp
    assert weid == [-1, 7, -1, 7, 9, -1, -1, -1], weid
    # cửa sổ tràn bắt đầu giữa từ 5
    tnp, weid = segboot_token_columns([None, 5, 5, 6, None], {5: 1, 6: 2})
    assert tnp == [-1, 1, 1, 3, -1] and weid == [-1, 1, -1, 2, -1]
    print("ok  token columns")


def test_closure_vs_unionfind():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = 40
        A = rng.random((T, T)) < 0.04
        A = A | A.T | np.eye(T, dtype=bool)
        parent = list(range(T))
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(T):
            for j in range(T):
                if A[i, j]:
                    parent[f(i)] = f(j)
        ref = np.array([[f(i) == f(j) for j in range(T)] for i in range(T)])
        got = transitive_closure(torch.tensor(A)[None])[0].numpy()
        assert (ref == got).all()
    print("ok  transitive closure == union-find")


def test_component_boxes_ids():
    wb = torch.tensor([[[10, 10, 20, 20], [30, 12, 40, 22], [100, 100, 110, 110]]], dtype=torch.float)
    node = torch.tensor([[True, True, True]])
    C = torch.tensor([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]], dtype=torch.bool)
    cb = component_boxes(C, wb, node)
    assert cb[0, 0].tolist() == [10, 10, 40, 22] and cb[0, 2].tolist() == [100, 100, 110, 110]
    assert component_ids(C, node)[0].tolist() == [0, 0, 2]
    print("ok  component boxes / ids")


def test_forward_backward_and_no_leakage():
    torch.manual_seed(0)
    m = Model()
    batch, _, _ = build_batch(synthetic_doc())
    m.train()
    out = m(**batch)
    assert torch.isfinite(out["loss"]), out["loss"]
    out["loss"].backward()
    assert m.bb.emb.weight.grad is not None and m.bb.emb.weight.grad.abs().sum() > 0
    assert m.segboot.group_head.W.grad is not None
    assert m.segboot.type_head.vis_proj.weight.grad is not None
    # eval: đầu ra không phụ thuộc word_entity_id / labels
    m.eval()
    with torch.no_grad():
        a = m(**batch)
        b2 = dict(batch); b2["word_entity_id"] = torch.randint(0, 3, batch["word_entity_id"].shape)
        b2["labels"] = None
        b = m(**b2)
        b3 = dict(batch); b3.pop("word_entity_id"); b3.pop("labels")
        c = m(**b3)
    for k in ("logits", "pred_groups", "pred_types"):
        assert torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]), k
    print("ok  forward/backward, eval không rò rỉ nhãn")


def test_decode_perfect_groups_metric():
    """Nhóm gold + loại gold -> BIO khớp retag & cả hai metric = 1.0 (kể cả khi thứ tự bị xáo)."""
    words = synthetic_doc(seed=3)
    core = SegBootCore(H, ID2LABEL)
    for order in (list(range(len(words))), perturb_order(list(range(len(words))), random.Random(1), prob=1.0)):
        batch, orig_wid, labs_orig = build_batch(words, order)
        T = batch["input_ids"].shape[1]
        node = batch["word_start"].bool()
        eid = batch["word_entity_id"]
        C = (eid[:, :, None] == eid[:, None, :]) & node[:, :, None] & node[:, None, :]
        cid = component_ids(C, node)
        lab = batch["labels"][0]
        ctype = torch.zeros(1, T, dtype=torch.long)
        for t in range(T):
            if lab[t] >= 0:
                ctype[0, t] = int(core.bio2type[lab[t]])
        bio = core.bio_from_groups(cid, ctype, node)
        gold_bio = [LABELS[int(l)] for l in lab if l >= 0]
        pred_bio = [LABELS[int(bio[0, t])] for t in range(T) if lab[t] >= 0]
        assert pred_bio == gold_bio, (pred_bio, gold_bio)
        g = entity_set_prf_groups(torch.where(node, cid, -1).numpy(), torch.where(node, ctype, -1).numpy(),
                                  core.type_names, [orig_wid], [0], [labs_orig])
        assert abs(g["group_entity_f1"] - 1.0) < 1e-9, g
    print("ok  giải mã nhóm -> BIO khớp retag; group F1 = 1.0 với nhóm gold")


def test_overfit_sanity():
    """Học thuộc 4 tài liệu tổng hợp: group F1 trên train phải tiến tới ~1."""
    torch.manual_seed(0)
    m = Model()
    docs = [synthetic_doc(seed=s) for s in range(4)]
    batches = [build_batch(d, perturb_order(list(range(len(d))), random.Random(s), prob=1.0), seed=s)
               for s, d in enumerate(docs)]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    cb = SegBootEpsCallback(1.0, 0.0, 0.6)

    class St: max_steps = 300; global_step = 0
    st = St()
    for step in range(300):
        st.global_step = step
        cb.on_step_begin(None, st, None, model=m)
        m.train()
        b, _, _ = batches[step % 4]
        loss = m(**b)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    assert m.segboot.ss_eps == 0.0
    m.eval()
    f1s = []
    with torch.no_grad():
        for (b, ow, labs) in batches:
            o = m(**{k: v for k, v in b.items() if k not in ("labels", "word_entity_id")})
            r = entity_set_prf_groups(o["pred_groups"].numpy(), o["pred_types"].numpy(),
                                      m.segboot.type_names, [ow], [0], [labs])
            f1s.append(r["group_entity_f1"])
    print(f"    overfit group-F1 per doc: {[round(x, 3) for x in f1s]}  (loss cuối {loss.item():.4f})")
    assert np.mean(f1s) > 0.9, f1s
    print("ok  overfit sanity")


def test_unpack_and_prior():
    l, g, t = unpack_predictions((np.zeros((1, 3, 7)), np.zeros((1, 3)), np.zeros((1, 3))))
    assert g is not None and t is not None
    l, g, t = unpack_predictions(np.zeros((1, 3, 7)))
    assert g is None
    pr = type_prior_from_features([[0, 3, 4, -100, 5]], [[1, 1, 1, 0, 1]], LABELS)
    assert abs(sum(pr) - 1) < 1e-9 and len(pr) == 4
    print("ok  unpack predictions / type prior")


if __name__ == "__main__":
    test_token_columns()
    test_closure_vs_unionfind()
    test_component_boxes_ids()
    test_forward_backward_and_no_leakage()
    test_decode_perfect_groups_metric()
    test_unpack_and_prior()
    test_overfit_sanity()
    print("\nTẤT CẢ TEST ĐỀU PASS")
