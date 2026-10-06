# coding=utf-8
"""
LayoutLMv3ForSegBootV2 — LayoutLMv3 + residual segment layout tự bootstrap (xem segboot_v2_core.py).
Một encoder duy nhất (không còn grouper_encoder); head BIO giống hệt LayoutLMv3ForTokenClassification.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput

from .modeling_layoutlmv3 import LayoutLMv3ClassificationHead, LayoutLMv3Model, LayoutLMv3PreTrainedModel
from .segboot_v2_core import SegBootV2Core


@dataclass
class SegBootV2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class LayoutLMv3ForSegBootV2(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"segboot"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        if config.num_labels < 10:            # đúng như LayoutLMv3ForTokenClassification
            classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)
        self.segboot = SegBootV2Core(
            classifier, hidden=config.hidden_size, num_labels=config.num_labels,
            k_nn=int(getattr(config, "segboot_knn", 24)),
            tau=float(getattr(config, "segboot_tau", 0.5)),
            lambda_group=float(getattr(config, "segboot_lambda_group", 1.0)),
            lambda_aux=float(getattr(config, "segboot_lambda_aux", 0.5)),
            dropout=config.hidden_dropout_prob,
        )
        # Head BIO học với lr backbone như B0; group_head + seg_gate học với new_param_lr
        self.backbone_lr_keys = ("segboot.tok_classifier.",)
        self.init_weights()
        self.segboot.reset_parameters()

    def reset_new_parameters(self):
        """Gọi sau from_pretrained(base model) khi train (không gọi khi load checkpoint đã fine-tune)."""
        self.segboot.reset_parameters()

    def forward(self, input_ids=None, bbox=None, attention_mask=None, token_type_ids=None,
                position_ids=None, valid_span=None, head_mask=None, inputs_embeds=None,
                labels=None, images=None, word_start=None, token_node_pos=None,
                word_entity_id=None, line_ids=None, block_ids=None, column_ids=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **unused):
        if word_start is None or token_node_pos is None:
            raise ValueError("SegBoot cần word_start và token_node_pos.")
        T = input_ids.shape[1]
        emb = self.layoutlmv3.embeddings

        def run_encoder(embeds):
            out = self.layoutlmv3(input_ids, bbox=bbox, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, position_ids=position_ids,
                                  head_mask=head_mask, inputs_embeds=embeds,
                                  output_attentions=False, output_hidden_states=False, return_dict=True,
                                  images=images, valid_span=valid_span,
                                  line_ids=line_ids, block_ids=block_ids, column_ids=column_ids)
            return out[0][:, :T]

        res = self.segboot(run_encoder, emb.word_embeddings, emb._calc_spatial_position_embeddings,
                           input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                           word_start=word_start, token_node_pos=token_node_pos, labels=labels,
                           word_entity_id=word_entity_id,
                           special_ids=(self.config.bos_token_id, self.config.eos_token_id,
                                        self.config.pad_token_id))
        return SegBootV2Output(loss=res["loss"], logits=res["logits"])