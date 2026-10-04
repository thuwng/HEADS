# coding=utf-8
"""
segboot_core.py — phần lõi của SegBoot, KHÔNG phụ thuộc backbone (để unit test được).

Luồng xử lý (một cửa sổ 512 token):
  Pass 1 : backbone(word boxes)              -> h1
           PairGroupHead(h1) trên k-NN không gian -> logit "cùng entity" G1
           giải mã: ngưỡng + bao đóng bắc cầu   -> các nhóm C1 (không phụ thuộc thứ tự đọc)
  Boxes  : box mỗi từ := union box của nhóm chứa nó
           (huấn luyện: scheduled sampling giữa nhóm GOLD và nhóm dự đoán)
  Pass 2 : backbone(segment boxes tự dự đoán) -> h2      (cùng trọng số)
           PairGroupHead(h2) -> G2 -> nhóm cuối C2
  Typing : [h2_i ; mean-pool h2 theo nhóm ; RoIAlign ảnh theo box nhóm ; hình học nhóm]
           -> loại (O / HEADER / QUESTION / ANSWER ...), lấy trung bình logit trong nhóm.
  Output : logits BIO "one-hot" theo thứ tự chuỗi (để dùng nguyên seqeval / entity_set_prf cũ)
           + pred_groups (id nhóm) + pred_types (để chấm entity-set chính xác).

Ghi chú quan trọng:
  * Embedding toạ độ của LayoutLMv3 là bảng tra số nguyên -> box của pass 2 là RỜI RẠC,
    không có gradient qua box. Gradient đi vào backbone qua loss của pass 1 (grouping)
    và pass 2 (grouping + typing). Đây là lý do dùng scheduled sampling thay vì "soft box".
  * word_entity_id CHỈ dùng cho TARGET và scheduled sampling khi model.training=True.
    Ở eval (model.eval()) đầu ra độc lập hoàn toàn với word_entity_id và labels
    (có unit test kiểm tra).
"""

import math
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


# =============================================================================
# Tiện ích
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
    """adj: (B,T,T) bool, đối xứng, có self-loop. Trả về ma trận liên thông (bool).
    Bình phương lặp: tối đa ceil(log2 T) phép nhân ma trận."""
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
    """C: (B,T,T) bool membership; word_box: (B,T,4) float [0,1000]; node: (B,T) bool.
    Trả về (B,T,4): union box của nhóm chứa từng từ (non-node giữ box gốc)."""
    big = 1e6
    Cm = C & node[:, None, :]
    wb = word_box[:, None, :, :]                       # (B,1,T,4)
    x0 = torch.where(Cm, wb[..., 0], torch.full_like(wb[..., 0], big)).amin(-1)
    y0 = torch.where(Cm, wb[..., 1], torch.full_like(wb[..., 1], big)).amin(-1)
    x1 = torch.where(Cm, wb[..., 2], torch.full_like(wb[..., 2], -big)).amax(-1)
    y1 = torch.where(Cm, wb[..., 3], torch.full_like(wb[..., 3], -big)).amax(-1)
    box = torch.stack([x0, y0, x1, y1], dim=-1)
    has = Cm.any(-1, keepdim=True)
    return torch.where(has, box, word_box)


def component_ids(C: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
    """id nhóm = vị trí token nhỏ nhất trong nhóm; -1 với non-node."""
    B, T, _ = C.shape
    ar = torch.arange(T, device=C.device).view(1, 1, T).expand(B, T, T)
    cid = torch.where(C & node[:, None, :], ar, torch.full_like(ar, T)).amin(-1)
    return torch.where(node, cid, torch.full_like(cid, -1))


def knn_pair_mask(word_box: torch.Tensor, node: torch.Tensor, k: int, y_scale: float = 2.0) -> torch.Tensor:
    """Mặt nạ cặp (i,j) nằm trong k láng giềng gần nhất (đối xứng hoá), chỉ giữa các node."""
    B, T, _ = word_box.shape
    cx = (word_box[..., 0] + word_box[..., 2]) * 0.5
    cy = (word_box[..., 1] + word_box[..., 3]) * 0.5
    dx = cx[:, :, None] - cx[:, None, :]
    dy = (cy[:, :, None] - cy[:, None, :]) * y_scale
    d = torch.sqrt(dx * dx + dy * dy)
    pair = node[:, :, None] & node[:, None, :]
    d = torch.where(pair, d, torch.full_like(d, float("inf")))
    k_eff = max(1, min(k, T))
    idx = d.topk(k_eff, dim=-1, largest=False).indices
    M = torch.zeros(B, T, T, dtype=torch.bool, device=word_box.device)
    M.scatter_(-1, idx, True)
    M = (M | M.transpose(1, 2)) & pair
    return M


def pair_geometry_features(box: torch.Tensor) -> torch.Tensor:
    """box: (B,T,4) đã chia 1000 về [0,1]. Trả về (B,T,T,8)."""
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
    # nhân hệ số để các đặc trưng có độ lớn tương đương
    return torch.stack([dcx * 10, dcy * 10, gap_x * 10, gap_y * 10, ov_y, ov_x, dleft * 10, log_hr], dim=-1)


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    t = target.float()
    ce = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none")
    p = torch.sigmoid(logits.float())
    pt = p * t + (1 - p) * (1 - t)
    return (ce * (1 - pt).pow(gamma)).mean()


# =============================================================================
# Các head
# =============================================================================

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
            self.bias.fill_(-2.0)        # đa số cặp k-NN là "khác entity"

    def forward(self, h: torch.Tensor, box01: torch.Tensor) -> torch.Tensor:
        a, b = self.a(h), self.b(h)
        s = torch.matmul(torch.matmul(a, self.W), b.transpose(1, 2)) / math.sqrt(self.d)
        g = self.geo(pair_geometry_features(box01)).squeeze(-1)
        s = s.float() + g.float() + self.bias
        return 0.5 * (s + s.transpose(1, 2))


class RegionTypeHead(nn.Module):
    """Phân loại theo nhóm: token + pooled nhóm + RoI thị giác + hình học nhóm."""

    def __init__(self, hidden: int, num_types: int, roi_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.roi_size = roi_size
        self.vis_proj = nn.Linear(hidden * roi_size * roi_size, hidden)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3 + 6, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_types),
        )

    def roi_features(self, image_h: Optional[torch.Tensor], boxes: torch.Tensor) -> torch.Tensor:
        B, T, _ = boxes.shape
        H = self.vis_proj.out_features
        if image_h is None or image_h.shape[1] < 2 or roi_align is None:
            return boxes.new_zeros(B, T, H)
        n = image_h.shape[1] - 1                     # bỏ CLS ảnh
        g = int(round(math.sqrt(n)))
        if g * g != n:
            return boxes.new_zeros(B, T, H)
        fmap = image_h[:, 1:].transpose(1, 2).reshape(B, -1, g, g).float()
        bidx = torch.arange(B, device=boxes.device, dtype=torch.float).view(B, 1, 1).expand(B, T, 1)
        rois = torch.cat([bidx, boxes.float()], dim=-1).reshape(-1, 5)
        out = roi_align(fmap, rois, output_size=self.roi_size, spatial_scale=g / 1000.0,
                        sampling_ratio=2, aligned=True)                   # (B*T, H, r, r)
        out = out.reshape(B, T, -1).to(self.vis_proj.weight.dtype)
        return self.vis_proj(out)

    def forward(self, h, seg_h, image_h, comp_box, page_ref_h):
        vis = self.roi_features(image_h, comp_box).to(h.dtype)
        b = comp_box.float() / 1000.0
        w = (b[..., 2] - b[..., 0]).clamp(min=0)
        hh = (b[..., 3] - b[..., 1]).clamp(min=0)
        cx = (b[..., 0] + b[..., 2]) * 0.5
        cy = (b[..., 1] + b[..., 3]) * 0.5
        # chiều cao tương đối so với chiều cao từ trung vị (xấp xỉ cỡ chữ) – tín hiệu cho HEADER
        rel_h = hh / page_ref_h.clamp(min=1e-3).view(-1, 1)          # page_ref_h: (B,)
        geo = torch.stack([w, hh, cx, cy, torch.log1p(rel_h), (cx - 0.5).abs()], dim=-1).to(h.dtype)
        return self.mlp(torch.cat([h, seg_h, vis, geo], dim=-1))


# =============================================================================
# Lõi SegBoot
# =============================================================================

class SegBootCore(nn.Module):
    def __init__(self, hidden: int, id2label: dict, k_nn: int = 24, tau: float = 0.5,
                 lambda_group: float = 1.0, logit_adj: float = 1.0,
                 type_prior: Optional[Sequence[float]] = None, dropout: float = 0.1,
                 final_groups: str = "pass2"):
        super().__init__()
        labels = [id2label[i] for i in range(len(id2label))]
        types = ["O"] + sorted({l[2:] for l in labels if l != "O"})
        self.type_names = types
        K = len(types)
        label2id = {l: i for i, l in enumerate(labels)}
        o_id = label2id["O"]
        bio2type = [types.index("O") if l == "O" else types.index(l[2:]) for l in labels]
        b_ids = [o_id] + [label2id["B-" + t] for t in types[1:]]
        i_ids = [o_id] + [label2id["I-" + t] for t in types[1:]]
        self.register_buffer("bio2type", torch.tensor(bio2type, dtype=torch.long), persistent=False)
        self.register_buffer("type2B", torch.tensor(b_ids, dtype=torch.long), persistent=False)
        self.register_buffer("type2I", torch.tensor(i_ids, dtype=torch.long), persistent=False)
        prior = torch.tensor(type_prior if type_prior is not None else [1.0] * K, dtype=torch.float)
        prior = prior / prior.sum()
        self.register_buffer("log_prior", prior.clamp(min=1e-6).log(), persistent=False)

        self.num_labels = len(labels)
        self.o_id = o_id
        self.k_nn, self.tau = k_nn, tau
        self.lambda_group, self.logit_adj = lambda_group, logit_adj
        assert final_groups in ("pass1", "pass2")
        self.final_groups = final_groups

        self.group_head = PairGroupHead(hidden, dropout=dropout)
        self.type_head = RegionTypeHead(hidden, K, dropout=dropout)

        # Được cập nhật bởi callback: xác suất dùng nhóm GOLD cho pass 2 khi train.
        self.ss_eps = 1.0
        # Chỉ dùng cho phân tích "upper bound": pass 2 dùng nhóm gold cả ở test. KHÔNG báo cáo như kết quả chính.
        self.eval_oracle = False

    def reset_parameters(self):
        self.group_head.reset_parameters()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_groups(self, G: torch.Tensor, knn: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        B, T, _ = G.shape
        eye = torch.eye(T, dtype=torch.bool, device=G.device).unsqueeze(0)
        adj = (torch.sigmoid(G) > self.tau) & knn
        adj = adj | (eye & node[:, :, None])
        return transitive_closure(adj) & node[:, :, None] & node[:, None, :]

    def bio_from_groups(self, cid: torch.Tensor, ctype: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """Chuyển (nhóm, loại) thành nhãn BIO theo thứ tự chuỗi hiện tại.
        Quy tắc giống retag_bio_after_reorder: I- nếu từ liền trước (theo chuỗi) cùng nhóm."""
        B, T = cid.shape
        pos = torch.arange(T, device=cid.device).expand(B, T)
        node_pos = torch.where(node, pos, torch.full_like(pos, -1))
        shifted = torch.cat([torch.full_like(node_pos[:, :1], -1), node_pos[:, :-1]], dim=1)
        prev = torch.cummax(shifted, dim=1).values
        prev_cid = cid.gather(1, prev.clamp(min=0))
        is_I = node & (prev >= 0) & (prev_cid == cid) & (ctype != 0)
        lab = torch.where(is_I, self.type2I[ctype], self.type2B[ctype])
        return torch.where(node, lab, torch.full_like(lab, self.o_id))

    # ------------------------------------------------------------------
    def forward(
        self,
        run_backbone: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        word_start: torch.Tensor,
        token_node_pos: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        word_entity_id: Optional[torch.Tensor] = None,
        special_ids: Sequence[int] = (),
    ):
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
        wbox = bbox[:, :T].float()
        box01 = wbox / 1000.0
        knn = knn_pair_mask(wbox, node, self.k_nn)
        eye = torch.eye(T, dtype=torch.bool, device=dev).unsqueeze(0)
        pair_mask = knn & ~eye

        # ---------------- gold (chỉ TARGET / scheduled sampling) ----------------
        gold_C = None
        if word_entity_id is not None:
            eid = fit_len(word_entity_id, T, -1).long()
            gn = node & (eid >= 0)
            gold_C = (eid[:, :, None] == eid[:, None, :]) & gn[:, :, None] & gn[:, None, :]
            gold_C = gold_C | (eye & node[:, :, None])

        # ---------------- pass 1 ----------------
        h1, _ = run_backbone(bbox)
        G1 = self.group_head(h1, box01)
        C1 = self.decode_groups(G1, knn, node)

        if self.training and gold_C is not None:
            use_gold = torch.rand(B, device=dev) < float(self.ss_eps)
        elif (not self.training) and self.eval_oracle and gold_C is not None:
            use_gold = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            use_gold = torch.zeros(B, dtype=torch.bool, device=dev)
        C_in = torch.where(use_gold[:, None, None], gold_C, C1) if gold_C is not None else C1

        # ---------------- bootstrapped segment boxes ----------------
        cbox = component_boxes(C_in, wbox, node)
        tok_box = cbox.gather(1, tnp.clamp(min=0).unsqueeze(-1).expand(B, T, 4))
        new_text_box = torch.where((tnp >= 0).unsqueeze(-1), tok_box, wbox)
        bbox2 = bbox.clone()
        bbox2[:, :T] = new_text_box.round().clamp(0, 1000).to(bbox.dtype)

        # ---------------- pass 2 ----------------
        h2, img2 = run_backbone(bbox2)
        G2 = self.group_head(h2, box01)
        C2 = self.decode_groups(G2, knn, node)
        C_final = C2 if self.final_groups == "pass2" else C1

        if gold_C is not None and (self.training or self.eval_oracle):
            C_type = torch.where(use_gold[:, None, None], gold_C, C_final)
        else:
            C_type = C_final

        Mf = (C_type & node[:, None, :]).to(h2.dtype)
        Mn = Mf / Mf.sum(-1, keepdim=True).clamp(min=1.0)
        seg_h = torch.bmm(Mn, h2)
        tbox = component_boxes(C_type, wbox, node)
        word_h = (wbox[..., 3] - wbox[..., 1]).clamp(min=1.0) / 1000.0
        ref_h = torch.stack([word_h[b][node[b]].median() if node[b].any() else word_h.new_tensor(0.01)
                             for b in range(B)])
        type_logits = self.type_head(h2, seg_h, img2, tbox, ref_h).float()      # (B,T,K)

        # ---------------- giải mã ----------------
        comp_logits = torch.bmm(Mn.float(), type_logits)
        ctype = comp_logits.argmax(-1)
        cid = component_ids(C_type, node)
        bio = self.bio_from_groups(cid, ctype, node)
        out_logits = torch.full((B, T, self.num_labels), -10.0, device=dev)
        out_logits.scatter_(-1, bio.unsqueeze(-1), 10.0)
        pred_groups = torch.where(node, cid, torch.full_like(cid, -1))
        pred_types = torch.where(node, ctype, torch.full_like(ctype, -1))

        # ---------------- loss ----------------
        loss = None
        if labels is not None and gold_C is not None:
            lg = focal_bce(G1[pair_mask], gold_C[pair_mask]) + focal_bce(G2[pair_mask], gold_C[pair_mask])
            lab = fit_len(labels[:, :T], T, -100)
            tmask = node & (lab != -100)
            ttarget = self.bio2type[lab.clamp(min=0)]
            adj_logits = type_logits + self.logit_adj * self.log_prior     # logit adjustment (Menon et al.)
            lt = F.cross_entropy(adj_logits[tmask], ttarget[tmask]) if tmask.any() else type_logits.sum() * 0.0
            loss = self.lambda_group * lg + lt

        return {
            "loss": loss,
            "logits": out_logits,
            "pred_groups": pred_groups,
            "pred_types": pred_types,
        }
