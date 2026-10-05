# coding=utf-8
"""
segboot_v2_core.py — SegBoot v2 (sửa các lỗi thiết kế của v1, xem HUONG_DAN_V2.md).

Khác biệt cốt lõi so với v1:
  (1) ĐẦU RA = head BIO theo token của LayoutLMv3 (dropout + Linear + CE), y hệt baseline
      đạt 90.2 ở setting A. Phần nhóm KHÔNG còn quyết định entity cuối cùng; nó chỉ sinh
      segment box cho bộ gán nhãn. -> giải mã và metric giống hệt LayoutLMv3.
  (2) Giải mã nhóm "ưu tiên precision": ngưỡng cao (mặc định 0.7) + BCE thường (hiệu chỉnh tốt
      hơn focal). Tách nhầm -> box gần với word box (lùi về hành vi B0-B, vô hại);
      gộp nhầm -> box sai (có hại). Vì vậy ta chấp nhận tách hơn là gộp.
  (3) Tuỳ chọn hai encoder: grouper (word box) và tagger (segment box). Tagger chỉ bao giờ
      thấy segment box như mô hình ở setting A -> upper bound = 90.2 khi nhóm hoàn hảo.
  (4) Scheduled sampling không giảm về 0 (eps_end mặc định 0.3) để tagger luôn thấy cả box
      sạch lẫn box nhiễu.
  (5) Thống kê chất lượng phân đoạn (pairwise P/R/F1, tỷ lệ segment khớp đúng) được tích luỹ
      ở eval khi có word_entity_id — CHỈ để ghi log, không ảnh hưởng đầu ra.
"""

from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segboot_core import (
    PairGroupHead, component_boxes, fit_len, focal_bce, knn_pair_mask, transitive_closure,
)


class SegBootV2Core(nn.Module):
    def __init__(self, hidden: int, num_labels: int, k_nn: int = 24, tau: float = 0.7,
                 lambda_group: float = 1.0, lambda_aux: float = 0.0, focal_gamma: float = 0.0,
                 dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.k_nn, self.tau = k_nn, tau
        self.lambda_group, self.lambda_aux = lambda_group, lambda_aux
        self.focal_gamma = focal_gamma
        self.group_head = PairGroupHead(hidden, dropout=dropout)
        # Head BIO giống LayoutLMv3ForTokenClassification (num_labels < 10 -> Linear)
        self.dropout = nn.Dropout(dropout)
        self.tok_classifier = nn.Linear(hidden, num_labels)
        self.ss_eps = 1.0
        self.eval_oracle = False
        self.reset_seg_stats()

    def reset_parameters(self):
        self.group_head.reset_parameters()

    # ---------------- thống kê phân đoạn (chỉ log) ----------------
    def reset_seg_stats(self):
        self.seg_stats = {"pair_tp": 0, "pair_fp": 0, "pair_fn": 0, "seg_exact": 0, "seg_total": 0}

    def pop_seg_stats(self, prefix: str = "seg_"):
        s = self.seg_stats
        P = s["pair_tp"] / max(1, s["pair_tp"] + s["pair_fp"])
        R = s["pair_tp"] / max(1, s["pair_tp"] + s["pair_fn"])
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        out = {f"{prefix}pair_precision": P, f"{prefix}pair_recall": R, f"{prefix}pair_f1": F1,
               f"{prefix}exact_rate": s["seg_exact"] / max(1, s["seg_total"])}
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
        row_eq = ((C_pred == C_gold) | ~node[:, None, :]).all(-1) | ~node       # (B,T)
        ar = torch.arange(T, device=C_pred.device).view(1, 1, T).expand(B, T, T)
        gcid = torch.where(C_gold & node[:, None, :], ar, torch.full_like(ar, T)).amin(-1)
        for b in range(B):
            nb = node[b]
            if not nb.any():
                continue
            ids = gcid[b][nb]
            ok = row_eq[b][nb]
            uniq = torch.unique(ids)
            self.seg_stats["seg_total"] += int(uniq.numel())
            bad = torch.unique(ids[~ok]) if (~ok).any() else ids.new_empty(0)
            self.seg_stats["seg_exact"] += int(uniq.numel() - bad.numel())

    # ---------------- forward ----------------
    def forward(
        self,
        run_grouper: Callable[[torch.Tensor], torch.Tensor],
        run_tagger: Callable[[torch.Tensor], torch.Tensor],
        input_ids, bbox, attention_mask, word_start, token_node_pos,
        labels=None, word_entity_id=None, special_ids: Sequence[int] = (),
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
        knn = knn_pair_mask(wbox, node, self.k_nn)
        eye = torch.eye(T, dtype=torch.bool, device=dev).unsqueeze(0)
        pair_mask = knn & ~eye

        gold_C = None
        if word_entity_id is not None:
            eid = fit_len(word_entity_id, T, -1).long()
            gn = node & (eid >= 0)
            gold_C = (eid[:, :, None] == eid[:, None, :]) & gn[:, :, None] & gn[:, None, :]
            gold_C = gold_C | (eye & node[:, :, None])

        # ---- grouper (word box) ----
        h1 = run_grouper(bbox)
        G = self.group_head(h1, wbox / 1000.0)
        with torch.no_grad():
            adj = (torch.sigmoid(G) > self.tau) & knn
            adj = adj | (eye & node[:, :, None])
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

        # ---- segment boxes -> tagger ----
        cbox = component_boxes(C_in, wbox, node)
        tok_box = cbox.gather(1, tnp.clamp(min=0).unsqueeze(-1).expand(B, T, 4))
        new_text_box = torch.where((tnp >= 0).unsqueeze(-1), tok_box, wbox)
        bbox2 = bbox.clone()
        bbox2[:, :T] = new_text_box.round().clamp(0, 1000).to(bbox.dtype)
        h2 = run_tagger(bbox2)
        logits = self.tok_classifier(self.dropout(h2))                     # (B,T,C)

        loss = None
        if labels is not None:
            lab = fit_len(labels[:, :T], T, -100)
            loss = F.cross_entropy(logits.reshape(-1, self.num_labels).float(), lab.reshape(-1),
                                   ignore_index=-100)
            if gold_C is not None and self.lambda_group > 0:
                if self.focal_gamma > 0:
                    lg = focal_bce(G[pair_mask], gold_C[pair_mask], gamma=self.focal_gamma)
                else:
                    lg = F.binary_cross_entropy_with_logits(G[pair_mask].float(), gold_C[pair_mask].float()) \
                        if pair_mask.any() else G.sum() * 0.0
                loss = loss + self.lambda_group * lg
            if self.lambda_aux > 0:                     # chỉ có ý nghĩa khi grouper dùng chung encoder
                aux = self.tok_classifier(self.dropout(h1))
                loss = loss + self.lambda_aux * F.cross_entropy(
                    aux.reshape(-1, self.num_labels).float(), lab.reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}
