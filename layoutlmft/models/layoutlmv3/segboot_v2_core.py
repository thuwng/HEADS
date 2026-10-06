# coding=utf-8
"""
segboot_v2_core.py — SegBoot (bản sửa): residual segment layout trên MỘT encoder LayoutLMv3.

  Pass 1 (= LayoutLMv3 gốc): encoder(word box) -> h1
         -> head BIO (loss phụ, lambda_aux)  +  PairGroupHead trên k-NN -> nhóm dự đoán C
  Segment box: box của mỗi từ := union box của nhóm chứa nó
         (train: scheduled sampling giữa nhóm GOLD và nhóm dự đoán, eps giảm về 0)
  Pass 2: CÙNG encoder. 2D position embedding và 2D attention bias vẫn dùng WORD box;
         text embedding += g * (Spatial(segment box) - Spatial(word box)),  g khởi tạo = 0
         -> head BIO -> đầu ra cuối (giải mã + seqeval y hệt LayoutLMv3).

Tính chất:
  * Lúc khởi tạo pass 2 == pass 1 == LayoutLMv3 (B0). Thông tin segment chỉ được dùng ở mức
    gradient cho thấy là có lợi, nên nhóm sai không còn ép tagger đặt B- tại mỗi chỗ đổi box.
  * Một encoder duy nhất -> số tham số ~ B0 (+ group head ~0.4M + gate 768).
  * Ở eval, đầu ra không phụ thuộc word_entity_id / labels (xem self-test cuối file).
"""

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Tiện ích (chuyển từ segboot_core.py để có thể xoá file đó)
# =============================================================================

def fit_len(x: Optional[torch.Tensor], length: int, pad_value) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.shape[1] == length:
        return x
    if x.shape[1] > length:
        return x[:, :length]
    pad = torch.full((x.shape[0], length - x.shape[1]) + tuple(x.shape[2:]),
                     pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def transitive_closure(adj: torch.Tensor) -> torch.Tensor:
    """adj: (B,T,T) bool, đối xứng, có self-loop. Trả về ma trận liên thông (bool)."""
    T = adj.shape[-1]
    dev_type = adj.device.type if adj.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        A = adj.float()
        for _ in range(max(1, math.ceil(math.log2(max(T, 2)))) + 1):
            A_new = (torch.bmm(A, A) > 0).float()
            if torch.equal(A_new, A):
                break
            A = A_new
    return A > 0


def component_boxes(C: torch.Tensor, word_box: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
    """C: (B,T,T) bool; word_box: (B,T,4) float [0,1000]; node: (B,T) bool -> (B,T,4) union box nhóm."""
    big = 1e6
    Cm = C & node[:, None, :]
    wb = word_box[:, None, :, :]
    x0 = torch.where(Cm, wb[..., 0], torch.full_like(wb[..., 0], big)).amin(-1)
    y0 = torch.where(Cm, wb[..., 1], torch.full_like(wb[..., 1], big)).amin(-1)
    x1 = torch.where(Cm, wb[..., 2], torch.full_like(wb[..., 2], -big)).amax(-1)
    y1 = torch.where(Cm, wb[..., 3], torch.full_like(wb[..., 3], -big)).amax(-1)
    box = torch.stack([x0, y0, x1, y1], dim=-1)
    has = Cm.any(-1, keepdim=True)
    return torch.where(has, box, word_box)


def knn_pair_mask(word_box: torch.Tensor, node: torch.Tensor, k: int, y_scale: float = 2.0) -> torch.Tensor:
    """Cặp (i,j) nằm trong k láng giềng gần nhất (đối xứng hoá), chỉ giữa các node."""
    B, T, _ = word_box.shape
    cx = (word_box[..., 0] + word_box[..., 2]) * 0.5
    cy = (word_box[..., 1] + word_box[..., 3]) * 0.5
    dx = cx[:, :, None] - cx[:, None, :]
    dy = (cy[:, :, None] - cy[:, None, :]) * y_scale
    d = torch.sqrt(dx * dx + dy * dy)
    pair = node[:, :, None] & node[:, None, :]
    d = torch.where(pair, d, torch.full_like(d, float("inf")))
    idx = d.topk(max(1, min(k, T)), dim=-1, largest=False).indices
    M = torch.zeros(B, T, T, dtype=torch.bool, device=word_box.device)
    M.scatter_(-1, idx, True)
    return (M | M.transpose(1, 2)) & pair


def pair_geometry_features(box: torch.Tensor) -> torch.Tensor:
    """box: (B,T,4) trong [0,1]. Trả về (B,T,T,8)."""
    x0, y0, x1, y1 = box.unbind(-1)
    w = (x1 - x0).clamp(min=1e-3)
    h = (y1 - y0).clamp(min=1e-3)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    dcx = cx[:, None, :] - cx[:, :, None]
    dcy = cy[:, None, :] - cy[:, :, None]
    gap_x = torch.maximum(x0[:, None, :] - x1[:, :, None], x0[:, :, None] - x1[:, None, :]).clamp(min=0)
    gap_y = torch.maximum(y0[:, None, :] - y1[:, :, None], y0[:, :, None] - y1[:, None, :]).clamp(min=0)
    ov_y = (torch.minimum(y1[:, None, :], y1[:, :, None]) - torch.maximum(y0[:, None, :], y0[:, :, None])).clamp(min=0)
    ov_y = ov_y / torch.minimum(h[:, None, :], h[:, :, None])
    ov_x = (torch.minimum(x1[:, None, :], x1[:, :, None]) - torch.maximum(x0[:, None, :], x0[:, :, None])).clamp(min=0)
    ov_x = ov_x / torch.minimum(w[:, None, :], w[:, :, None])
    dleft = x0[:, None, :] - x0[:, :, None]
    log_hr = torch.log(h[:, None, :] / h[:, :, None])
    return torch.stack([dcx * 10, dcy * 10, gap_x * 10, gap_y * 10, ov_y, ov_x, dleft * 10, log_hr], dim=-1)


class PairGroupHead(nn.Module):
    """Biaffine + hình học tương đối -> logit 'cùng entity' cho mọi cặp (đối xứng)."""

    def __init__(self, hidden: int, d: int = 256, geo_hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.a = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, d), nn.GELU())
        self.b = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, d), nn.GELU())
        self.W = nn.Parameter(torch.zeros(d, d))
        self.geo = nn.Sequential(nn.Linear(8, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.d = d

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.W, std=0.02)
            self.W.add_(torch.eye(self.d) * 0.1)
            self.bias.fill_(-2.0)

    def forward(self, h: torch.Tensor, box01: torch.Tensor) -> torch.Tensor:
        a, b = self.a(h), self.b(h)
        s = torch.matmul(torch.matmul(a, self.W), b.transpose(1, 2)) / math.sqrt(self.d)
        g = self.geo(pair_geometry_features(box01)).squeeze(-1)
        s = s.float() + g.float() + self.bias
        return 0.5 * (s + s.transpose(1, 2))


# =============================================================================
# Lõi SegBoot
# =============================================================================

class SegBootV2Core(nn.Module):
    def __init__(self, classifier: nn.Module, hidden: int, num_labels: int, k_nn: int = 24,
                 tau: float = 0.5, lambda_group: float = 1.0, lambda_aux: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.k_nn, self.tau = k_nn, tau
        self.lambda_group, self.lambda_aux = lambda_group, lambda_aux
        self.group_head = PairGroupHead(hidden, dropout=dropout)
        # Head BIO giống hệt LayoutLMv3ForTokenClassification (Linear nếu <10 nhãn, MLP nếu không)
        self.dropout = nn.Dropout(dropout)
        self.tok_classifier = classifier
        # Cổng theo chiều cho phần dư segment-box; =0 -> pass 2 trùng LayoutLMv3 gốc
        self.seg_gate = nn.Parameter(torch.zeros(hidden))
        self.ss_eps = 1.0          # cập nhật bởi SegBootEpsCallback
        self.eval_oracle = False   # CHỈ để phân tích upper bound
        self.reset_seg_stats()

    def reset_parameters(self):
        self.group_head.reset_parameters()
        with torch.no_grad():
            self.seg_gate.zero_()

    # ---------------- thống kê phân đoạn (chỉ log) ----------------
    def reset_seg_stats(self):
        self.seg_stats = {"pair_tp": 0, "pair_fp": 0, "pair_fn": 0, "seg_exact": 0, "seg_total": 0}

    def pop_seg_stats(self, prefix: str = "seg_"):
        s = self.seg_stats
        P = s["pair_tp"] / max(1, s["pair_tp"] + s["pair_fp"])
        R = s["pair_tp"] / max(1, s["pair_tp"] + s["pair_fn"])
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        out = {f"{prefix}pair_precision": P, f"{prefix}pair_recall": R, f"{prefix}pair_f1": F1,
               f"{prefix}exact_rate": s["seg_exact"] / max(1, s["seg_total"]),
               f"{prefix}gate_abs_mean": float(self.seg_gate.detach().abs().mean())}
        self.reset_seg_stats()
        return out

    @torch.no_grad()
    def _accumulate_stats(self, C_pred, C_gold, node):
        B, T, _ = C_pred.shape
        eye = torch.eye(T, dtype=torch.bool, device=C_pred.device).unsqueeze(0)
        valid = node[:, :, None] & node[:, None, :] & ~eye
        self.seg_stats["pair_tp"] += int((C_pred & C_gold & valid).sum()) // 2
        self.seg_stats["pair_fp"] += int((C_pred & ~C_gold & valid).sum()) // 2
        self.seg_stats["pair_fn"] += int((~C_pred & C_gold & valid).sum()) // 2
        row_eq = ((C_pred == C_gold) | ~node[:, None, :]).all(-1) | ~node
        ar = torch.arange(T, device=C_pred.device).view(1, 1, T).expand(B, T, T)
        gcid = torch.where(C_gold & node[:, None, :], ar, torch.full_like(ar, T)).amin(-1)
        for b in range(B):
            nb = node[b]
            if not nb.any():
                continue
            ids, ok = gcid[b][nb], row_eq[b][nb]
            uniq = torch.unique(ids)
            self.seg_stats["seg_total"] += int(uniq.numel())
            bad = torch.unique(ids[~ok]) if (~ok).any() else ids.new_empty(0)
            self.seg_stats["seg_exact"] += int(uniq.numel() - bad.numel())

    # ---------------- forward ----------------
    def forward(
        self,
        run_encoder: Callable[[Optional[torch.Tensor]], torch.Tensor],
        embed_words: Callable[[torch.Tensor], torch.Tensor],
        embed_spatial: Callable[[torch.Tensor], torch.Tensor],
        input_ids, bbox, attention_mask, word_start, token_node_pos,
        labels=None, word_entity_id=None, special_ids: Sequence[int] = (),
    ):
        """run_encoder(inputs_embeds|None) -> hidden text (B,T,H), luôn với WORD box."""
        B, T = input_ids.shape
        dev = input_ids.device
        valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        if attention_mask is not None:
            valid &= attention_mask[:, :T].bool()
        for tid in special_ids:
            if tid is not None:
                valid &= input_ids != tid
        node = fit_len(word_start, T, 0).bool() & valid
        tnp = fit_len(token_node_pos, T, -1).long()
        tbox = bbox[:, :T]
        wbox = tbox.float()
        knn = knn_pair_mask(wbox, node, self.k_nn)
        eye = torch.eye(T, dtype=torch.bool, device=dev).unsqueeze(0)
        pair_mask = knn & ~eye

        gold_C = None
        if word_entity_id is not None:
            eid = fit_len(word_entity_id, T, -1).long()
            gn = node & (eid >= 0)
            gold_C = (eid[:, :, None] == eid[:, None, :]) & gn[:, :, None] & gn[:, None, :]
            gold_C = gold_C | (eye & node[:, :, None])

        # ---- pass 1: LayoutLMv3 gốc trên word box ----
        h1 = run_encoder(None)
        logits1 = self.tok_classifier(self.dropout(h1))
        G = self.group_head(h1, wbox / 1000.0)
        with torch.no_grad():
            adj = ((torch.sigmoid(G) > self.tau) & knn) | (eye & node[:, :, None])
            C_pred = transitive_closure(adj) & node[:, :, None] & node[:, None, :]

        if self.training and gold_C is not None:
            use_gold = torch.rand(B, device=dev) < float(self.ss_eps)
        elif (not self.training) and self.eval_oracle and gold_C is not None:
            use_gold = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            use_gold = torch.zeros(B, dtype=torch.bool, device=dev)
        C_in = torch.where(use_gold[:, None, None], gold_C, C_pred) if gold_C is not None else C_pred

        if (not self.training) and gold_C is not None:
            self._accumulate_stats(C_pred, gold_C, node)

        # ---- segment box tự dự đoán ----
        cbox = component_boxes(C_in, wbox, node)
        tok_box = cbox.gather(1, tnp.clamp(min=0).unsqueeze(-1).expand(B, T, 4))
        seg_box = torch.where((tnp >= 0).unsqueeze(-1), tok_box, wbox)
        seg_box = seg_box.round().clamp(0, 1000).to(tbox.dtype)

        # ---- pass 2: cùng encoder, word box giữ nguyên, cộng phần dư segment có cổng ----
        delta = embed_spatial(seg_box) - embed_spatial(tbox)              # =0 ở token đặc biệt
        inputs_embeds = embed_words(input_ids) + self.seg_gate.to(delta.dtype) * delta
        h2 = run_encoder(inputs_embeds)
        logits = self.tok_classifier(self.dropout(h2))

        loss = None
        if labels is not None:
            lab = fit_len(labels[:, :T], T, -100).reshape(-1)

            def ce(lg):
                return F.cross_entropy(lg.reshape(-1, self.num_labels).float(), lab, ignore_index=-100)

            loss = ce(logits)
            if self.lambda_aux > 0:
                loss = loss + self.lambda_aux * ce(logits1)
            if gold_C is not None and self.lambda_group > 0 and pair_mask.any():
                loss = loss + self.lambda_group * F.binary_cross_entropy_with_logits(
                    G[pair_mask].float(), gold_C[pair_mask].float())
        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    # Self-test (CPU, không cần backbone): python -m layoutlmft.models.layoutlmv3.segboot_v2_core
    torch.manual_seed(0)
    B, T, H, C = 2, 12, 32, 7
    word_emb, spat = nn.Embedding(50, H), nn.Linear(4, H)
    enc = nn.Linear(H, H)

    def embed_spatial(b):
        return spat(b.float() / 1000.0)

    ids = torch.randint(3, 50, (B, T)); ids[:, 0] = 0; ids[:, -1] = 2
    box = torch.randint(0, 900, (B, T, 4)); box[..., 2:] = box[..., :2] + 50
    box[:, 0] = 0; box[:, -1] = 0
    ws = torch.ones(B, T, dtype=torch.long); ws[:, 0] = 0; ws[:, -1] = 0
    tnp = torch.arange(T).expand(B, T).clone(); tnp[:, 0] = -1; tnp[:, -1] = -1
    eid = torch.randint(0, 4, (B, T)); eid[:, 0] = -1; eid[:, -1] = -1
    lab = torch.randint(0, C, (B, T)); lab[:, 0] = -100; lab[:, -1] = -100

    def run_encoder(ie):
        x = word_emb(ids) if ie is None else ie
        return enc(x + embed_spatial(box))

    core = SegBootV2Core(nn.Linear(H, C), H, C, k_nn=4)
    core.reset_parameters()
    core.eval()
    with torch.no_grad():
        a = core(run_encoder, word_emb, embed_spatial, ids, box, None, ws, tnp)["logits"]
        b = core(run_encoder, word_emb, embed_spatial, ids, box, None, ws, tnp,
                 labels=lab, word_entity_id=eid)["logits"]
        ref = core.tok_classifier(run_encoder(None))
    assert torch.allclose(a, b), "eval phụ thuộc nhãn gold!"
    assert torch.allclose(a, ref, atol=1e-6), "gate=0 nhưng pass 2 khác LayoutLMv3 gốc!"
    core.train(); core.ss_eps = 1.0   # dùng nhóm gold -> delta != 0
    out = core(run_encoder, word_emb, embed_spatial, ids, box, None, ws, tnp, labels=lab, word_entity_id=eid)
    out["loss"].backward()
    assert core.seg_gate.grad is not None and core.seg_gate.grad.abs().sum() > 0, "gate không nhận gradient"
    print("segboot_v2_core self-test passed")