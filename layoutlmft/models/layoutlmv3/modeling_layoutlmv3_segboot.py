# coding=utf-8
"""
LayoutLMv3ForSegBootKIE — SegBoot: self-bootstrapped segment layout + order-invariant grouping.

Đặt file này cạnh modeling_layoutlmv3.py. Phần thuật toán nằm trong segboot_core.py.

Đầu vào bắt buộc (thêm bởi tokenize_and_align_labels, xem HUONG_DAN_SUA.md):
  word_start      (B,T) 1 tại sub-token đầu của mỗi từ trong cửa sổ
  token_node_pos  (B,T) vị trí sub-token đầu của từ chứa token t (-1 với CLS/SEP/PAD)
  word_entity_id  (B,T) id entity GOLD tại sub-token đầu (-1 nơi khác)   [chỉ TARGET]
Yêu cầu: bbox đầu vào phải là WORD-LEVEL (setting B/C). Ở setting A box đã là oracle.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.file_utils import ModelOutput

from .modeling_layoutlmv3 import LayoutLMv3Model, LayoutLMv3PreTrainedModel
from .segboot_core import SegBootCore


@dataclass
class SegBootOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pred_groups: torch.LongTensor = None
    pred_types: torch.LongTensor = None


class LayoutLMv3ForSegBootKIE(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.segboot = SegBootCore(
            hidden=config.hidden_size,
            id2label={int(k): v for k, v in config.id2label.items()},
            k_nn=int(getattr(config, "segboot_knn", 24)),
            tau=float(getattr(config, "segboot_tau", 0.5)),
            lambda_group=float(getattr(config, "segboot_lambda_group", 1.0)),
            logit_adj=float(getattr(config, "segboot_logit_adj", 1.0)),
            type_prior=getattr(config, "segboot_type_prior", None),
            dropout=config.hidden_dropout_prob,
            final_groups=str(getattr(config, "segboot_final_groups", "pass2")),
        )
        self.init_weights()
        self.reset_new_parameters()

    def reset_new_parameters(self):
        """Gọi lại sau from_pretrained() (giống LatentSegment) vì _init_weights có thể ghi đè."""
        self.segboot.reset_parameters()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        images=None,
        word_start=None,
        token_node_pos=None,
        word_entity_id=None,
        line_ids=None,
        block_ids=None,
        column_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **unused,
    ):
        if word_start is None or token_node_pos is None:
            raise ValueError("SegBoot cần word_start và token_node_pos (xem HUONG_DAN_SUA.md, bước 3).")
        T = input_ids.shape[1]

        def run_backbone(bb) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            out = self.layoutlmv3(
                input_ids, bbox=bb, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                output_attentions=False, output_hidden_states=False, return_dict=True,
                images=images, valid_span=valid_span,
                line_ids=line_ids, block_ids=block_ids, column_ids=column_ids,
            )
            seq = out[0]
            img = seq[:, T:] if seq.shape[1] > T else None
            return seq[:, :T], img

        res = self.segboot(
            run_backbone, input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
            word_start=word_start, token_node_pos=token_node_pos,
            labels=labels, word_entity_id=word_entity_id,
            special_ids=(self.config.bos_token_id, self.config.eos_token_id, self.config.pad_token_id),
        )
        return SegBootOutput(
            loss=res["loss"], logits=res["logits"],
            pred_groups=res["pred_groups"], pred_types=res["pred_types"],
        )
