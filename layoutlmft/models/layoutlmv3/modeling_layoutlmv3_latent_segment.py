# coding=utf-8
"""
LayoutLMv3ForLatentSegmentKIE  —  oracle-free, differentiable segmentation head.

Ý tưởng chính (continuous relaxation của hard segment mean-pooling hiện tại):

  1. Boundary predictor:   p_t = sigmoid(f([h_{t-1}; h_t; geo(t-1,t)]))
                           (chỉ được tách tại token đầu của một từ)
  2. Boundary energy:      e_t = -log(1 - p_t) = softplus(logit_t)
                           c_t = sum_{k<=t} e_k
  3. Soft membership:      s_ij = prod_{k=i+1..j}(1 - p_k) = exp(-|c_i - c_j|)
                           A = softmax_j(-|c_i - c_j|)  (chỉ trên token hợp lệ)
                           z_i = sum_j A_ij h_j
     -> Khi p in {0,1}, z_i trùng CHÍNH XÁC với hard mean-pooling theo segment.
  4. Segment-head attention: logit_ij += log p_j  (mỗi soft-segment được đại diện
     bởi token đầu của nó) + 2D relative-geometry bias giữa tâm các soft-segment.
  5. Token/segment gate theo từng token, thay cho alpha cố định <= 0.15.
  6. Soft start cue: p_t * E[1] + (1 - p_t) * E[0]  (thay is_first_token_embedding
     vốn = nhãn B- khi seg_id lấy từ segment-level bbox).

Huấn luyện: CE + lambda_b * BCE(boundary)  (+ tùy chọn order-consistency KL).
Boundary target chỉ dựng từ nhãn TRAIN, không bao giờ là input của forward.

YÊU CẦU DỮ LIỆU (quan trọng để tránh leakage):
  - bbox phải là WORD-LEVEL (không dùng segment-level box của FUNSD/CORD),
    nếu không geo(t-1,t) sẽ lộ ranh giới entity.
  - Dataset cung cấp thêm cột: word_start (0/1), boundary_labels (0/1/-100),
    orig_word_id (>=0 ở sub-token đầu, -1 nơi khác) nếu dùng consistency loss.
    Data collator phải pad các cột này (pad: 0 / -100 / -1).

LƯU Ý: code này chưa được chạy trong môi trường của tôi; hãy chạy unit test
(xem cuối file) trước khi huấn luyện.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)

NEG = -1e4  # finite mask value (an toàn cho fp16/fp32, tránh NaN khi cả hàng bị mask)


# =============================================================================
# 1. DATA SIDE (pure Python) – gọi trong tokenize_and_align_labels
# =============================================================================

def boundary_targets_from_word_labels(word_labels: List[str]) -> List[int]:
    """
    1 nếu từ thứ t mở đầu một entity span mới, 0 nếu tiếp nối span trước.
    Chuỗi O liên tiếp được coi là một span (FUNSD không phân biệt các entity
    'other' kề nhau trong nhãn BIO).
    """
    targets, prev_type = [], None
    for i, lab in enumerate(word_labels):
        if lab == "O":
            cur_type, is_b = "O", False
        else:
            prefix, cur_type = lab.split("-", 1)
            is_b = prefix in ("B", "S")
        targets.append(int(i == 0 or is_b or cur_type != prev_type))
        prev_type = cur_type
    return targets


def align_word_level_columns(word_ids, word_boundary, order=None):
    """
    word_ids: tokenized_inputs.word_ids(batch_index) (index theo thứ tự đã reorder)
    word_boundary: output của boundary_targets_from_word_labels (theo thứ tự đã reorder)
    order: list ánh xạ vị trí-sau-reorder -> index từ gốc (None nếu không reorder)
    """
    word_start, boundary_labels, orig_word_id = [], [], []
    prev = None
    for w in word_ids:
        if w is None:
            word_start.append(0); boundary_labels.append(-100); orig_word_id.append(-1)
        elif w != prev:
            word_start.append(1)
            boundary_labels.append(int(word_boundary[w]))
            orig_word_id.append(int(order[w]) if order is not None else int(w))
        else:
            word_start.append(0); boundary_labels.append(-100); orig_word_id.append(-1)
        prev = w
    return word_start, boundary_labels, orig_word_id


# =============================================================================
# 2. GEOMETRY HELPERS
# =============================================================================

def pair_geometry(bbox: torch.Tensor) -> torch.Tensor:
    """Đặc trưng hình học giữa token t-1 và t theo thứ tự đọc. bbox: (B, L, 4) in [0,1000]."""
    b = bbox.float()
    p = torch.roll(b, shifts=1, dims=1)
    h_b = (b[..., 3] - b[..., 1]).clamp(min=1.0)
    h_p = (p[..., 3] - p[..., 1]).clamp(min=1.0)
    w_b = (b[..., 2] - b[..., 0]).clamp(min=1.0)
    w_p = (p[..., 2] - p[..., 0]).clamp(min=1.0)
    h = torch.maximum(h_b, h_p)

    gap_x = (b[..., 0] - p[..., 2]) / h
    d_top = (b[..., 1] - p[..., 1]) / h
    d_cy = ((b[..., 1] + b[..., 3]) - (p[..., 1] + p[..., 3])) / (2.0 * h)
    log_hr = torch.log(h_b / h_p)
    inter_y = (torch.minimum(b[..., 3], p[..., 3]) - torch.maximum(b[..., 1], p[..., 1])).clamp(min=0)
    inter_x = (torch.minimum(b[..., 2], p[..., 2]) - torch.maximum(b[..., 0], p[..., 0])).clamp(min=0)
    ov_y = inter_y / torch.minimum(h_b, h_p)
    ov_x = inter_x / torch.minimum(w_b, w_p)

    feats = torch.stack([gap_x, d_top, d_cy, log_hr, ov_y, ov_x], dim=-1).clamp(-10.0, 10.0)
    feats = torch.cat([torch.zeros_like(feats[:, :1]), feats[:, 1:]], dim=1)  # t=0 không có t-1
    return feats


def signed_log_bucket(d: torch.Tensor, num_buckets: int, max_dist: float = 1000.0) -> torch.Tensor:
    v = torch.sign(d) * torch.log1p(d.abs()) / math.log1p(max_dist)
    v = v.clamp(-1.0, 1.0)
    return ((v + 1.0) * 0.5 * (num_buckets - 1)).round().long()


# =============================================================================
# 3. MODULES
# =============================================================================

class SoftSegmentPooling(nn.Module):
    GEO_DIM = 6

    def __init__(self, hidden_size: int, window: int = None):
        super().__init__()
        inner = max(64, hidden_size // 4)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + self.GEO_DIM, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        self.window = window

    def forward(self, h, bbox, valid, word_start):
        """
        h: (B,L,H); bbox: (B,L,4); valid, word_start: (B,L) bool
        returns z (B,L,H), A (B,L,L) float32, boundary_logit (B,L),
                log_head (B,L) [bias cho attention], head_prob (B,L)
        """
        B, L, _ = h.shape
        h_prev = torch.roll(h, shifts=1, dims=1)
        geo = pair_geometry(bbox).to(h.dtype)
        logit = self.boundary_mlp(torch.cat([h_prev, h, geo], dim=-1)).squeeze(-1).float()

        prev_valid = torch.roll(valid, shifts=1, dims=1)
        prev_valid[:, 0] = False
        first_valid = valid & ~prev_valid
        can_split = valid & word_start & ~first_valid

        # e_t = -log(1 - sigmoid(logit_t)) = softplus(logit_t); sub-token tiếp nối: e_t = 0
        energy = torch.where(can_split, F.softplus(logit), torch.zeros_like(logit))
        c = torch.cumsum(energy, dim=1)                                  # (B,L), float32
        scores = -(c.unsqueeze(2) - c.unsqueeze(1)).abs()               # log s_ij

        key_ok = valid.unsqueeze(1).expand(B, L, L)
        if self.window is not None:
            idx = torch.arange(L, device=h.device)
            band = (idx[None, :] - idx[:, None]).abs() <= self.window
            key_ok = key_ok & band.unsqueeze(0)
        scores = scores.masked_fill(~key_ok, NEG)
        A = torch.softmax(scores, dim=-1)                                # soft membership
        z = torch.matmul(A.to(h.dtype), h)

        log_head = F.logsigmoid(logit)
        log_head = torch.where(first_valid, torch.zeros_like(log_head), log_head)
        log_head = log_head.masked_fill(~(can_split | first_valid), NEG)
        head_prob = torch.where(can_split, torch.sigmoid(logit), first_valid.float())
        return z, A, logit, log_head, head_prob


class GeoRelationalSegmentContext(nn.Module):
    """Attention giữa các soft-segment: key được trọng số bởi log p_j, cộng 2D relative bias."""

    def __init__(self, hidden_size: int, num_heads: int = 4, num_buckets: int = 33, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("num_heads must divide hidden_size")
        self.nh, self.dh = num_heads, hidden_size // num_heads
        self.num_buckets = num_buckets
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_size, hidden_size),
        )
        self.bias_x = nn.Embedding(num_buckets, num_heads)
        self.bias_y = nn.Embedding(num_buckets, num_heads)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))  # ReZero như bản cũ

    def forward(self, z, A, bbox, valid, log_head):
        B, L, H = z.shape
        b = bbox.float()
        centers = torch.stack([(b[..., 0] + b[..., 2]) * 0.5, (b[..., 1] + b[..., 3]) * 0.5], dim=-1)
        seg_c = torch.matmul(A, centers)                                  # tâm soft-segment (B,L,2)
        dx = seg_c[..., 0].unsqueeze(1) - seg_c[..., 0].unsqueeze(2)      # dx[b,i,j] = x_j - x_i
        dy = seg_c[..., 1].unsqueeze(1) - seg_c[..., 1].unsqueeze(2)
        geo = self.bias_x(signed_log_bucket(dx, self.num_buckets)) \
            + self.bias_y(signed_log_bucket(dy, self.num_buckets))       # (B,L,L,nh)
        geo = geo.permute(0, 3, 1, 2)

        q, k, v = self.qkv(z).view(B, L, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)).float() / math.sqrt(self.dh)
        scores = scores + geo.float() + log_head[:, None, None, :]
        scores = scores.masked_fill(~valid[:, None, None, :], NEG)
        attn = self.drop(torch.softmax(scores, dim=-1)).to(v.dtype)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H)

        u = self.ln1(z + self.drop(self.out(ctx)))
        u = self.ln2(u + self.drop(self.ffn(u)))
        return z + self.gate * (u - z)


class TokenSegmentGate(nn.Module):
    """g_i in (0,1) theo từng token: g->1 dùng segment, g->0 khôi phục token (segment không thuần)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(2 * hidden_size, 1)

    def forward(self, h, s):
        g = torch.sigmoid(self.proj(torch.cat([h, s], dim=-1)))
        return h + g * (s - h), g.squeeze(-1)


def order_consistency_loss(logits_a, wid_a, logits_b, wid_b, temperature: float = 1.0):
    """
    Symmetric KL giữa 2 cách tuần tự hóa (vd XY-cut vs raster) của CÙNG tài liệu.
    logits_*: (B,T,C) phần text; wid_*: (B,T) orig_word_id (-1 nơi bỏ qua).
    Chỉ tính trên các từ xuất hiện ở cả hai view (do truncation có thể khác nhau).
    """
    B, _, C = logits_a.shape
    W = int(max(wid_a.max().item(), wid_b.max().item())) + 1
    if W <= 0:
        return logits_a.new_zeros(())

    def scatter(logits, wid):
        out = logits.new_zeros(B, W, C)
        present = torch.zeros(B, W, dtype=torch.bool, device=logits.device)
        bi, ti = torch.nonzero(wid >= 0, as_tuple=True)
        out[bi, wid[bi, ti]] = logits[bi, ti]
        present[bi, wid[bi, ti]] = True
        return out, present

    la, pa = scatter(logits_a, wid_a)
    lb, pb = scatter(logits_b, wid_b)
    m = pa & pb
    if not m.any():
        return logits_a.new_zeros(())
    lpa = F.log_softmax(la[m].float() / temperature, dim=-1)
    lpb = F.log_softmax(lb[m].float() / temperature, dim=-1)
    kl_ab = F.kl_div(lpa, lpb, log_target=True, reduction="batchmean")
    kl_ba = F.kl_div(lpb, lpa, log_target=True, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba) * temperature ** 2


# =============================================================================
# 4. MODEL
# =============================================================================

def _fit_len(x, length, pad_value):
    if x is None:
        return None
    if x.shape[1] == length:
        return x
    if x.shape[1] > length:
        return x[:, :length]
    pad = torch.full((x.shape[0], length - x.shape[1]), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


class LayoutLMv3ForLatentSegmentKIE(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        H = config.hidden_size
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(H, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        self.seg_pool = SoftSegmentPooling(H, window=getattr(config, "seg_window", None))
        self.seg_ctx = GeoRelationalSegmentContext(
            H, num_heads=int(getattr(config, "segment_context_heads", 4)),
            dropout=float(getattr(config, "segment_context_dropout", config.hidden_dropout_prob)),
        )
        self.gate = TokenSegmentGate(H)
        self.start_embedding = nn.Embedding(2, H)
        self.lambda_boundary = float(getattr(config, "lambda_boundary", 0.5))

        self.init_weights()
        self.reset_new_parameters()

    def reset_new_parameters(self):
        """
        Gọi LẠI hàm này sau from_pretrained(): một số phiên bản transformers
        khởi tạo lại các module 'missing' bằng _init_weights, ghi đè init đặc biệt.
        """
        with torch.no_grad():
            self.seg_ctx.gate.zero_()
            self.seg_ctx.bias_x.weight.zero_()
            self.seg_ctx.bias_y.weight.zero_()
            self.gate.proj.weight.zero_()
            self.gate.proj.bias.fill_(2.0)          # g≈0.88: khởi đầu segment-dominant
            last = self.seg_pool.boundary_mlp[-1]
            last.weight.zero_()
            last.bias.zero_()                       # p=0.5: làm mượt cục bộ 2^{-k} lúc đầu
            self.start_embedding.weight.normal_(0.0, 0.02)

    def forward(
        self, input_ids=None, bbox=None, attention_mask=None, token_type_ids=None,
        position_ids=None, valid_span=None, head_mask=None, inputs_embeds=None,
        labels=None, word_start=None, boundary_labels=None, orig_word_id=None,
        line_ids=None, block_ids=None, column_ids=None,
        output_attentions=None, output_hidden_states=None, return_dict=None, images=None,
        **unused,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.layoutlmv3(
            input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, images=images, valid_span=valid_span,
            line_ids=line_ids, block_ids=block_ids, column_ids=column_ids,
        )
        seq = outputs[0]
        T = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_h, image_h = seq[:, :T], seq[:, T:]

        valid = torch.ones(text_h.shape[:2], dtype=torch.bool, device=seq.device)
        if attention_mask is not None:
            valid &= attention_mask[:, :T].bool()
        if input_ids is not None:
            for tid in (self.config.bos_token_id, self.config.eos_token_id, self.config.pad_token_id):
                if tid is not None:
                    valid &= input_ids != tid
        ws = _fit_len(word_start, T, 0)
        ws = valid.clone() if ws is None else ws.bool()

        z, A, b_logit, log_head, head_prob = self.seg_pool(text_h, bbox[:, :T], valid, ws)
        ctx = self.seg_ctx(z, A, bbox[:, :T], valid, log_head)
        fused, g = self.gate(text_h, ctx)
        start_emb = head_prob.unsqueeze(-1).to(fused.dtype) * self.start_embedding.weight[1] \
            + (1.0 - head_prob).unsqueeze(-1).to(fused.dtype) * self.start_embedding.weight[0]
        logits_text = self.classifier(self.dropout(fused + start_emb))

        if image_h.shape[1] > 0:
            logits = torch.cat([logits_text, self.classifier(self.dropout(image_h))], dim=1)
        else:
            logits = logits_text

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            flat_logits = logits.reshape(-1, self.num_labels)
            flat_labels = labels.reshape(-1)
            if attention_mask is not None:
                active = attention_mask.reshape(-1) == 1
                flat_labels = torch.where(active, flat_labels, torch.full_like(flat_labels, -100))
            loss = loss_fct(flat_logits, flat_labels)

            bl = _fit_len(boundary_labels, T, -100)
            if bl is not None and self.lambda_boundary > 0:
                prev_valid = torch.roll(valid, shifts=1, dims=1)
                prev_valid[:, 0] = False
                m = (bl != -100) & valid & ws & prev_valid

        if not return_dict:
            out = (logits,) + outputs[2:]
            return ((loss,) + out) if loss is not None else out
        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


# =============================================================================
# 5. UNIT TESTS NÊN CHẠY (pytest) TRƯỚC KHI TRAIN
# =============================================================================
# (a) Hard-limit: ép logit = +50 tại ranh giới thật, -50 nơi khác  ->  z phải
#     khớp mean-pooling theo segment (torch.allclose, atol=1e-4).
# (b) Hàng A cộng bằng 1, A_ij = 0 (≈) với j là CLS/SEP/PAD.
# (c) Sub-token tiếp nối không bao giờ là head: head_prob == 0 tại đó.
# (d) Đổi boundary_labels không làm thay đổi logits (chỉ đổi loss) -> không leakage.
# (e) fp16 autocast: không NaN với batch có sample rỗng (toàn PAD).
