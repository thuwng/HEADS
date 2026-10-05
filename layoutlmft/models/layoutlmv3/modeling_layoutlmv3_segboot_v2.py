# coding=utf-8
"""
LayoutLMv3ForSegBootV2 — đầu ra BIO chuẩn LayoutLMv3; nhóm từ chỉ dùng để sinh segment box.

config.segboot_share_encoder:
  True  : một encoder chạy 2 lượt (word box rồi segment box) – ít tham số.
  False : grouper_encoder (word box) + layoutlmv3 (tagger, segment box) – mặc định khuyến nghị.
          grouper_encoder được khởi tạo bằng trọng số pretrained qua init_grouper_from_backbone().
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.file_utils import ModelOutput

from .modeling_layoutlmv3 import LayoutLMv3Model, LayoutLMv3PreTrainedModel
from .segboot_v2_core import SegBootV2Core


@dataclass
class SegBootV2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class LayoutLMv3ForSegBootV2(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"grouper_encoder", r"segboot"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.share = bool(getattr(config, "segboot_share_encoder", False))
        self.layoutlmv3 = LayoutLMv3Model(config)                    # tagger (nhận pretrained weights)
        self.grouper_encoder = None if self.share else LayoutLMv3Model(config)
        self.segboot = SegBootV2Core(
            hidden=config.hidden_size, num_labels=config.num_labels,
            k_nn=int(getattr(config, "segboot_knn", 24)),
            tau=float(getattr(config, "segboot_tau", 0.7)),
            lambda_group=float(getattr(config, "segboot_lambda_group", 1.0)),
            lambda_aux=float(getattr(config, "segboot_lambda_aux", 0.0)),
            focal_gamma=float(getattr(config, "segboot_focal_gamma", 0.0)),
            dropout=config.hidden_dropout_prob,
        )
        # Tham số học với learning rate của backbone (1e-5), không phải new_param_lr:
        self.backbone_lr_keys = ("grouper_encoder.", "segboot.tok_classifier.")
        self.init_weights()
        self.segboot.reset_parameters()

    def reset_new_parameters(self):
        self.segboot.reset_parameters()

    @torch.no_grad()
    def init_grouper_from_backbone(self):
        """Gọi MỘT LẦN sau from_pretrained(base model) khi bắt đầu train."""
        if self.grouper_encoder is not None:
            self.grouper_encoder.load_state_dict(self.layoutlmv3.state_dict())

    def forward(self, input_ids=None, bbox=None, attention_mask=None, token_type_ids=None,
                position_ids=None, valid_span=None, head_mask=None, inputs_embeds=None,
                labels=None, images=None, word_start=None, token_node_pos=None,
                word_entity_id=None, line_ids=None, block_ids=None, column_ids=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **unused):
        if word_start is None or token_node_pos is None:
            raise ValueError("SegBoot v2 cần word_start và token_node_pos.")
        T = input_ids.shape[1]

        def make_runner(enc):
            def run(bb):
                out = enc(input_ids, bbox=bb, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                          output_attentions=False, output_hidden_states=False, return_dict=True,
                          images=images, valid_span=valid_span,
                          line_ids=line_ids, block_ids=block_ids, column_ids=column_ids)
                return out[0][:, :T]
            return run

        run_tagger = make_runner(self.layoutlmv3)
        run_grouper = run_tagger if self.share else make_runner(self.grouper_encoder)
        res = self.segboot(run_grouper, run_tagger, input_ids=input_ids, bbox=bbox,
                           attention_mask=attention_mask, word_start=word_start,
                           token_node_pos=token_node_pos, labels=labels, word_entity_id=word_entity_id,
                           special_ids=(self.config.bos_token_id, self.config.eos_token_id,
                                        self.config.pad_token_id))
        return SegBootV2Output(loss=res["loss"], logits=res["logits"])
