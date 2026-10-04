#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
from layoutlmft.data.order_utils import retag_bio_after_reorder, entity_set_prf

import random
from layoutlmft.data.segboot_utils import (
    segboot_token_columns, perturb_order, make_dev_split, SegBootEpsCallback,
    type_prior_from_features, unpack_predictions, entity_set_prf_groups,
)

import transformers
import torch

from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from layoutlmft.data.image_utils import (
    RandomResizedCropAndInterpolationWithTwoPic,
    pil_loader,
    Compose,
)

from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_latent_segment import (
    boundary_targets_from_word_labels, align_word_level_columns,
)

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms


check_min_version("4.5.0")
logger = logging.getLogger(__name__)


# ==============================================================================
# ARGUMENTS
# ==============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

    use_hierarchical_position_encoding: bool = field(default=False)
    max_line_position: int = field(default=50)
    max_block_position: int = field(default=20)
    use_column_encoding: bool = field(default=False)
    max_column_position: int = field(default=10)

    use_intra_line_boundary: bool = field(default=False)
    lambda_bound_init: float = field(default=0.1)

    use_semantic_geometry_disentangle: bool = field(default=False)
    lambda_geo_init: float = field(default=0.1)
    lambda_orth_init: float = field(default=0.1)

    lambda_boundary: float = field(default=0.5)
    seg_window: Optional[int] = field(default=None)
    new_param_lr: float = field(default=5e-4)

    lds_use_ctx: bool = field(default=True)
    lds_use_gate: bool = field(default=True)
    lds_use_start_cue: bool = field(default=True)

    segboot_knn: int = field(default=24)
    segboot_tau: float = field(default=0.5)
    segboot_lambda_group: float = field(default=1.0)
    segboot_logit_adj: float = field(default=1.0)
    segboot_final_groups: str = field(default="pass2", metadata={"help": "pass1 | pass2"})
    segboot_eps_start: float = field(default=1.0)
    segboot_eps_end: float = field(default=0.0)
    segboot_eps_decay: float = field(default=0.6)
    segboot_eval_oracle: bool = field(default=False, metadata={"help": "CHỈ để phân tích upper bound"})

@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default="ner")
    dataset_name: Optional[str] = field(default="funsd")
    dataset_config_name: Optional[str] = field(default=None)

    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)

    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)

    max_train_samples: Optional[int] = field(default=None)
    max_val_samples: Optional[int] = field(default=None)
    max_test_samples: Optional[int] = field(default=None)

    label_all_tokens: bool = field(default=False)
    return_entity_level_metrics: bool = field(default=False)

    segment_level_layout: bool = field(default=True)
    bbox_level: str = field(default="word", metadata={"help": "word | segment (oracle)"})
    seg_source: str = field(default="line", metadata={"help": "line | oracle_bbox"})
    use_latent_segment: bool = field(default=False)
    visual_embed: bool = field(default=True)
    use_segment_head: bool = field(default=False)

    hipos_impl: str = field(
        default="legacy",
        metadata={"help": "legacy = line_ids theo y_center (bản 0,95) | visual = build_visual_lines"},
    )
    legacy_optimizer: bool = field(
        default=False,
        metadata={"help": "Tái hiện optimizer của bản 0,95: lr 5e-4 cho head, AdamW wd mặc định 0.01"},
    )

    # Automatic geometry-only reading-order / segmentation.
    # There is intentionally NO oracle-segment path in this file.
    apply_xy_cut: bool = field(
        default=True,
        metadata={"help": "Build reading order from OCR bounding boxes using recursive XY-Cut."},
    )

    data_dir: Optional[str] = field(default=None)

    input_size: int = field(default=224)
    second_input_size: int = field(default=112)

    train_interpolation: str = field(default="bicubic")
    second_interpolation: str = field(default="lanczos")
    imagenet_default_mean_and_std: bool = field(default=False)

    use_segboot: bool = field(default=False)
    eval_on: str = field(default="test", metadata={"help": "test (giống LayoutLMv3, chỉ dùng khi chạy final) | dev"})
    dev_ratio: float = field(default=0.2)
    dev_seed: int = field(default=42)
    dev_fold: int = field(default=-1)
    num_folds: int = field(default=5)
    order_aug_prob: float = field(default=0.0)

# ==============================================================================
# GEOMETRIC PRIMITIVES
# ==============================================================================

def _height(box):
    return max(1.0, float(box[3] - box[1]))


def _width(box):
    return max(1.0, float(box[2] - box[0]))


def _x_center(box):
    return 0.5 * (float(box[0]) + float(box[2]))


def _y_center(box):
    return 0.5 * (float(box[1]) + float(box[3]))


def _horizontal_gap(box_a, box_b):
    if float(box_a[2]) < float(box_b[0]):
        return float(box_b[0]) - float(box_a[2])
    if float(box_b[2]) < float(box_a[0]):
        return float(box_a[0]) - float(box_b[2])
    return 0.0


def _vertical_gap(box_a, box_b):
    if float(box_a[3]) < float(box_b[1]):
        return float(box_b[1]) - float(box_a[3])
    if float(box_b[3]) < float(box_a[1]):
        return float(box_a[1]) - float(box_b[3])
    return 0.0


def _vertical_overlap_ratio(box_a, box_b):
    top = max(float(box_a[1]), float(box_b[1]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, bottom - top)
    denom = max(1.0, min(_height(box_a), _height(box_b)))
    return inter / denom


def _horizontal_overlap_ratio(box_a, box_b):
    left = max(float(box_a[0]), float(box_b[0]))
    right = min(float(box_a[2]), float(box_b[2]))
    inter = max(0.0, right - left)
    denom = max(1.0, min(_width(box_a), _width(box_b)))
    return inter / denom


# ==============================================================================
# STEP 1: VISUAL LINE CONSTRUCTION
# ==============================================================================

def build_visual_lines(bboxes):
    """
    Build spatial text lines from OCR word boxes.

    IMPORTANT:
      - Uses only bounding boxes.
      - Does not inspect labels.
      - Does not use dataset-provided/oracle segments.

    A line is formed from words that are vertically aligned and close enough
    horizontally to plausibly belong to one visual text line.
    """
    n = len(bboxes)
    if n == 0:
        return []

    heights = np.asarray([_height(b) for b in bboxes], dtype=np.float32)
    median_h = max(1.0, float(np.median(heights)))

    # Adaptive thresholds.
    y_center_tol = 0.65 * median_h
    max_x_gap = 4.0 * median_h

    # Start from top-to-bottom, then left-to-right.
    order = sorted(
        range(n),
        key=lambda i: (_y_center(bboxes[i]), float(bboxes[i][0]), i),
    )

    lines = []

    for idx in order:
        box = bboxes[idx]
        yc = _y_center(box)

        best = None
        best_score = None

        for li, line in enumerate(lines):
            line_box = line["bbox"]

            dy = abs(yc - line["y_center"])
            overlap = _vertical_overlap_ratio(box, line_box)
            x_gap = _horizontal_gap(box, line_box)

            # Reject clearly different rows.
            if overlap < 0.20 and dy > y_center_tol:
                continue

            # Reject distant columns.
            if x_gap > max_x_gap:
                continue

            score = (
                dy / median_h
                + 0.20 * (x_gap / max_x_gap)
                - 1.50 * overlap
            )

            if best_score is None or score < best_score:
                best_score = score
                best = li

        if best is None:
            lines.append(
                {
                    "indices": [idx],
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "y_center": yc,
                }
            )
        else:
            line = lines[best]
            line["indices"].append(idx)
            line["bbox"] = [
                min(line["bbox"][0], float(box[0])),
                min(line["bbox"][1], float(box[1])),
                max(line["bbox"][2], float(box[2])),
                max(line["bbox"][3], float(box[3])),
            ]
            line["y_center"] = float(
                np.mean([_y_center(bboxes[j]) for j in line["indices"]])
            )

    # Always enforce left-to-right order inside each visual line.
    for line in lines:
        line["indices"].sort(
            key=lambda i: (float(bboxes[i][0]), _y_center(bboxes[i]), i)
        )

    # Temporary top-to-bottom ordering for deterministic region construction.
    lines.sort(
        key=lambda line: (
            float(line["bbox"][1]),
            float(line["bbox"][0]),
        )
    )

    return lines


# ==============================================================================
# STEP 2: XY-CUT ON LINE REGIONS (NOT ON INDIVIDUAL WORDS)
# ==============================================================================

def _largest_gap_split(regions, axis, median_h):
    """
    Find the largest meaningful whitespace split.

    axis=0 -> vertical partition (left/right), searching x gaps.
    axis=1 -> horizontal partition (top/bottom), searching y gaps.
    """
    if len(regions) <= 1:
        return None

    start = 0 if axis == 0 else 1
    end = 2 if axis == 0 else 3

    ordered = sorted(
        regions,
        key=lambda r: (
            float(r["bbox"][start]),
            float(r["bbox"][end]),
        ),
    )

    best_gap = -1.0
    best_index = -1
    running_end = float(ordered[0]["bbox"][end])

    for i in range(1, len(ordered)):
        current_start = float(ordered[i]["bbox"][start])
        gap = current_start - running_end

        if gap > best_gap:
            best_gap = gap
            best_index = i

        running_end = max(
            running_end,
            float(ordered[i]["bbox"][end]),
        )

    if best_index <= 0 or best_index >= len(ordered):
        return None

    # Normalise by text height so the algorithm is scale-robust.
    gap_ratio = best_gap / median_h

    # Vertical whitespace between adjacent text lines is usually small.
    # Column gaps are typically much wider.
    if axis == 0:
        min_ratio = 1.50
    else:
        min_ratio = 0.85

    if gap_ratio < min_ratio:
        return None

    return {
        "regions_a": ordered[:best_index],
        "regions_b": ordered[best_index:],
        "gap": best_gap,
        "score": gap_ratio,
        "axis": axis,
    }


def xy_cut_reading_order(lines, median_h):
    """
    Recursive XY-Cut over visual-line bounding regions.

    The recursion first looks for a genuine whitespace partition. At each
    node both x- and y-splits are considered; the strongest normalized gap is
    selected. Leaves are emitted top-to-bottom and left-to-right.
    """
    if len(lines) <= 1:
        return list(lines)

    candidates = []

    x_split = _largest_gap_split(
        lines,
        axis=0,
        median_h=median_h,
    )
    y_split = _largest_gap_split(
        lines,
        axis=1,
        median_h=median_h,
    )

    if x_split is not None:
        # Give a modest preference to a strong column gap.
        x_split["score"] *= 1.10
        candidates.append(x_split)

    if y_split is not None:
        candidates.append(y_split)

    if not candidates:
        return sorted(
            lines,
            key=lambda line: (
                float(line["bbox"][1]),
                float(line["bbox"][0]),
            ),
        )

    split = max(candidates, key=lambda x: x["score"])

    left = xy_cut_reading_order(
        split["regions_a"],
        median_h,
    )
    right = xy_cut_reading_order(
        split["regions_b"],
        median_h,
    )

    return left + right


# ==============================================================================
# STEP 3: BUILD AUTO SEGMENTS FROM THE READING-ORDERED VISUAL LINES
# ==============================================================================

def build_reading_order_and_segments(bboxes):
    """
    Geometry-only pipeline:

        word boxes
            -> visual lines
            -> XY-Cut reading order
            -> one visual line = one heuristic segment

    Returns:
        order: word indices in reading order
        seg_ids_orig: segment ID for every original word index

    No dataset-provided segment information is used.
    """
    if not bboxes:
        return [], []

    lines = build_visual_lines(bboxes)

    if not lines:
        return list(range(len(bboxes))), list(range(len(bboxes)))

    median_h = max(
        1.0,
        float(np.median([line["bbox"][3] - line["bbox"][1] for line in lines])),
    )

    ordered_lines = xy_cut_reading_order(
        lines,
        median_h,
    )

    # Flatten visual lines into the final reading order.
    order = []

    # Segment ID is defined by the ordered visual line.
    seg_ids_orig = [-1] * len(bboxes)

    for seg_id, line in enumerate(ordered_lines):
        for idx in line["indices"]:
            order.append(idx)
            seg_ids_orig[idx] = seg_id

    # Defensive consistency check.
    if len(order) != len(bboxes) or len(set(order)) != len(bboxes):
        order = list(range(len(bboxes)))

    for i, sid in enumerate(seg_ids_orig):
        if sid < 0:
            seg_ids_orig[i] = i

    return order, seg_ids_orig


# ==============================================================================
# OPTIONAL HIERARCHICAL POSITION IDS
# ==============================================================================

def compute_line_ids(bboxes):
    """
    Compute line IDs from the already reading-ordered boxes.

    IDs follow the same visual-line construction used for segmentation.
    """
    if not bboxes:
        return []

    lines = build_visual_lines(bboxes)
    line_map = [-1] * len(bboxes)

    # Sort by first occurrence in the current reading-order list.
    lines.sort(key=lambda line: min(line["indices"]))

    for lid, line in enumerate(lines):
        for idx in line["indices"]:
            line_map[idx] = lid

    for i in range(len(line_map)):
        if line_map[i] < 0:
            line_map[i] = i

    return line_map


def compute_block_ids(bboxes, x_threshold=50, y_threshold=30):
    """
    Conservative spatial block IDs on reading-ordered boxes.
    """
    if not bboxes:
        return []

    centers = [
        (_x_center(box), _y_center(box))
        for box in bboxes
    ]

    blocks = [0]
    current_block = 0

    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]

        if abs(dx) > x_threshold or abs(dy) > y_threshold:
            current_block += 1

        blocks.append(current_block)

    return blocks


def compute_column_ids(bboxes, x_threshold=50):
    """
    Column IDs estimated from x-centers in reading order.
    """
    if not bboxes:
        return []

    x_centers = [_x_center(box) for box in bboxes]
    columns = [0]
    current_col = 0

    for i in range(1, len(x_centers)):
        if abs(x_centers[i] - x_centers[i - 1]) > x_threshold:
            current_col += 1

        columns.append(current_col)

    return columns

def compute_line_ids_legacy(bboxes, y_threshold=10):
    """Đúng logic line_ids của bản 0,95 (định nghĩa thứ hai trong main cũ)."""
    if not bboxes:
        return []
    yc = [(b[1] + b[3]) / 2 for b in bboxes]
    out, cur = [0], 0
    for i in range(1, len(yc)):
        if abs(yc[i] - yc[i - 1]) > y_threshold:
            cur += 1
        out.append(cur)
    return out


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --------------------------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------------------------
    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(
            training_args.output_dir
        )

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the output_dir or add "
                "--overwrite_output_dir to train from scratch."
            )

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(
        logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Training/evaluation parameters {training_args}"
    )

    # --------------------------------------------------------------------------
    # Seed
    # --------------------------------------------------------------------------
    set_seed(training_args.seed)

    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    if data_args.dataset_name == "funsd":
        import layoutlmft.data.funsd

        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.funsd.__file__),
            cache_dir=model_args.cache_dir,
        )

    elif data_args.dataset_name == "cord":
        import layoutlmft.data.cord

        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.cord.__file__),
            cache_dir=model_args.cache_dir,
        )

    else:
        raise NotImplementedError(
            f"Unsupported dataset_name={data_args.dataset_name}. "
            "This runner supports FUNSD and CORD."
        )
    
    if data_args.eval_on == "dev" and "validation" not in datasets:
        datasets = make_dev_split(datasets, data_args.dev_ratio, data_args.dev_seed,
                                  data_args.dev_fold, data_args.num_folds)

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = (
        "words"
        if "words" in column_names
        else "tokens"
    )

    label_column_name = (
        f"{data_args.task_name}_tags"
        if f"{data_args.task_name}_tags" in column_names
        else column_names[1]
    )

    remove_columns = column_names

    if isinstance(
        features[label_column_name].feature,
        ClassLabel,
    ):
        label_list = features[label_column_name].feature.names
        label_to_id = {
            i: i
            for i in range(len(label_list))
        }
    else:
        unique_labels = set()

        for label_seq in datasets["train"][label_column_name]:
            unique_labels = unique_labels | set(label_seq)

        label_list = sorted(list(unique_labels))
        label_to_id = {
            label: i
            for i, label in enumerate(label_list)
        }

    num_labels = len(label_list)
    name_to_raw = {label_list[label_to_id[k]]: k for k in label_to_id}

    # --------------------------------------------------------------------------
    # Model config
    # --------------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=(
            True if model_args.use_auth_token else None
        ),
        use_hierarchical_position_encoding=(
            model_args.use_hierarchical_position_encoding
        ),
        max_line_position=model_args.max_line_position,
        max_block_position=model_args.max_block_position,
        use_column_encoding=model_args.use_column_encoding,
        max_column_position=model_args.max_column_position,
        use_intra_line_boundary=model_args.use_intra_line_boundary,
        lambda_bound_init=model_args.lambda_bound_init,
        use_semantic_geometry_disentangle=(
            model_args.use_semantic_geometry_disentangle
        ),
        lambda_geo_init=model_args.lambda_geo_init,
        lambda_orth_init=model_args.lambda_orth_init,

    )

    config.id2label = {i: l for i, l in enumerate(label_list)}
    config.label2id = {l: i for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        tokenizer_file=None,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=(
            True if model_args.use_auth_token else None
        ),
    )
    if data_args.use_segboot:
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segboot import LayoutLMv3ForSegBootKIE
        config.segboot_knn = model_args.segboot_knn
        config.segboot_tau = model_args.segboot_tau
        config.segboot_lambda_group = model_args.segboot_lambda_group
        config.segboot_logit_adj = model_args.segboot_logit_adj
        config.segboot_final_groups = model_args.segboot_final_groups
        model = LayoutLMv3ForSegBootKIE.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.reset_new_parameters()
        model.segboot.eval_oracle = model_args.segboot_eval_oracle
    elif data_args.use_latent_segment:
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_latent_segment import (
            LayoutLMv3ForLatentSegmentKIE,
        )
        config.lambda_boundary = model_args.lambda_boundary   # gán thẳng, không truyền qua from_pretrained
        config.seg_window = model_args.seg_window
        config.lds_use_ctx = model_args.lds_use_ctx
        config.lds_use_gate = model_args.lds_use_gate
        config.lds_use_start_cue = model_args.lds_use_start_cue
        model = LayoutLMv3ForLatentSegmentKIE.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.reset_new_parameters()
    elif getattr(data_args, "use_segment_head", False):
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
            LayoutLMv3ForSegmentTokenClassification,
        )

        model = (
            LayoutLMv3ForSegmentTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(
                    ".ckpt" in model_args.model_name_or_path
                ),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=(
                    True if model_args.use_auth_token else None
                ),
            )
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(
                ".ckpt" in model_args.model_name_or_path
            ),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=(
                True if model_args.use_auth_token else None
            ),
        )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer."
        )

    # --------------------------------------------------------------------------
    # Visual preprocessing
    # --------------------------------------------------------------------------
    padding = (
        "max_length"
        if data_args.pad_to_max_length
        else False
    )

    if data_args.visual_embed:
        imagenet_default_mean_and_std = (
            data_args.imagenet_default_mean_and_std
        )

        mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )

        std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )

        common_transform = Compose(
            [
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=data_args.input_size,
                    interpolation=data_args.train_interpolation,
                ),
            ]
        )

        patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std),
                ),
            ]
        )

    # --------------------------------------------------------------------------
    # Tokenization + geometry-only reading order + auto segmentation
    # --------------------------------------------------------------------------
    def tokenize_and_align_labels(examples, indices, augmentation=False, is_train=False):
        # Build the geometry-only reading order BEFORE tokenization.
        reordered_words = []
        reordered_bboxes = []
        reordered_labels = []
        reordered_orders = []
        reordered_eids = []

        for sample_idx in range(
            len(examples[text_column_name])
        ):
            words_i = examples[text_column_name][sample_idx]
            box_key = "bboxes_seg" if data_args.bbox_level == "segment" else "bboxes"
            bboxes_i = examples[box_key][sample_idx]
            labels_i = examples[label_column_name][sample_idx]

            if (
                getattr(data_args, "apply_xy_cut", True)
                and len(bboxes_i) > 1
            ):
                order, _ = build_reading_order_and_segments(
                    bboxes_i
                )
            else:
                order = list(range(len(bboxes_i)))
            if is_train and data_args.order_aug_prob > 0:
                rng = random.Random(training_args.seed * 100003 + int(indices[sample_idx]))
                order = perturb_order(order, rng, prob=data_args.order_aug_prob)
            eids_i = examples["entity_ids"][sample_idx] if "entity_ids" in examples else list(range(len(bboxes_i)))
            reordered_eids.append([eids_i[j] for j in order])

            reordered_orders.append(order)

            reordered_words.append(
                [words_i[j] for j in order]
            )
            reordered_bboxes.append(
                [bboxes_i[j] for j in order]
            )
            labels_str = [label_list[label_to_id[l]] for l in labels_i]
            new_str = retag_bio_after_reorder(labels_str, order)   # chỉ đụng TARGET
            if order == list(range(len(order))):
                assert new_str == labels_str, "retag làm đổi nhãn ở thứ tự gốc -> metric không còn giống LayoutLMv3"
            reordered_labels.append([name_to_raw[s] for s in new_str])

        tokenized_inputs = tokenizer(
            reordered_words,
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []

        seg_ids = []
        line_ids_all = []
        block_ids_all = []
        column_ids_all = []
        word_start_all, boundary_all, orig_wid_all = [], [], []
        token_node_pos_all, word_entity_all = [], []
        doc_idx_all = []

        for batch_index in range(
            len(tokenized_inputs["input_ids"])
        ):
            word_ids = tokenized_inputs.word_ids(
                batch_index=batch_index
            )

            original_batch_index = (
                tokenized_inputs[
                    "overflow_to_sample_mapping"
                ][batch_index]
            )

            label = reordered_labels[
                original_batch_index
            ]
            bbox = reordered_bboxes[
                original_batch_index
            ]
            words = reordered_words[
                original_batch_index
            ]

            # --------------------------------------------------------------
            # AUTO SEGMENTS: rebuild from the reordered word boxes.
            #
            # Absolutely no dataset-provided segment IDs are read here.
            # --------------------------------------------------------------
            word_seg_id = None
            if getattr(data_args, "use_segment_head", False):
                if data_args.seg_source == "oracle_bbox":
                    # ORACLE: chỉ dùng ở setting 1 để tái hiện baseline cũ
                    word_seg_id, cnt, prev = [], -1, None
                    for wb in bbox:
                        wb = tuple(wb)
                        if wb != prev:
                            cnt += 1
                            prev = wb
                        word_seg_id.append(cnt)
                else:
                    _, word_seg_id = build_reading_order_and_segments(bbox)

            line_ids_orig = (
                compute_line_ids_legacy(bbox)
                if data_args.hipos_impl == "legacy"
                else compute_line_ids(bbox)
            )
            block_ids_orig = compute_block_ids(bbox)
            column_ids_orig = compute_column_ids(bbox)

            word_label_strs = [label_list[label_to_id[l]] for l in label]
            ws_c, bl_c, ow_c = align_word_level_columns(
                word_ids, boundary_targets_from_word_labels(word_label_strs),
                order=reordered_orders[original_batch_index])
            word_start_all.append(ws_c); boundary_all.append(bl_c); orig_wid_all.append(ow_c)
            tnp_c, weid_c = segboot_token_columns(word_ids, reordered_eids[original_batch_index])
            token_node_pos_all.append(tnp_c); word_entity_all.append(weid_c)
            doc_idx_all.append(int(indices[original_batch_index]))

            previous_word_idx = None

            label_ids = []
            bbox_inputs = []
            seg_id_inputs = []
            line_ids_aligned = []
            block_ids_aligned = []
            column_ids_aligned = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append(
                        [0, 0, 0, 0]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(-1)

                    line_ids_aligned.append(-1)
                    block_ids_aligned.append(-1)
                    column_ids_aligned.append(-1)

                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label_to_id[
                            label[word_idx]
                        ]
                    )

                    bbox_inputs.append(
                        bbox[word_idx]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(
                            word_seg_id[word_idx]
                        )

                    line_ids_aligned.append(
                        line_ids_orig[word_idx]
                    )
                    block_ids_aligned.append(
                        block_ids_orig[word_idx]
                    )
                    column_ids_aligned.append(
                        column_ids_orig[word_idx]
                    )

                else:
                    label_ids.append(
                        label_to_id[
                            label[word_idx]
                        ]
                        if data_args.label_all_tokens
                        else -100
                    )

                    bbox_inputs.append(
                        bbox[word_idx]
                    )

                    if word_seg_id is not None:
                        seg_id_inputs.append(
                            word_seg_id[word_idx]
                        )

                    line_ids_aligned.append(
                        line_ids_orig[word_idx]
                    )
                    block_ids_aligned.append(
                        block_ids_orig[word_idx]
                    )
                    column_ids_aligned.append(
                        column_ids_orig[word_idx]
                    )

                previous_word_idx = word_idx

            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if word_seg_id is not None:
                seg_ids.append(seg_id_inputs)

            line_ids_all.append(line_ids_aligned)
            block_ids_all.append(block_ids_aligned)
            column_ids_all.append(column_ids_aligned)

            if data_args.visual_embed:
                ipath = examples["image_path"][
                    original_batch_index
                ]

                img = pil_loader(ipath)

                for_patches, _ = common_transform(
                    img,
                    augmentation=augmentation,
                )

                patch = patch_transform(
                    for_patches
                )

                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["line_ids"] = line_ids_all
        tokenized_inputs["block_ids"] = block_ids_all
        tokenized_inputs["column_ids"] = column_ids_all
        tokenized_inputs["word_start"] = word_start_all
        tokenized_inputs["boundary_labels"] = boundary_all
        tokenized_inputs["orig_word_id"] = orig_wid_all
        tokenized_inputs["doc_idx"] = doc_idx_all
        tokenized_inputs["token_node_pos"] = token_node_pos_all
        tokenized_inputs["word_entity_id"] = word_entity_all

        if getattr(
            data_args,
            "use_segment_head",
            False,
        ):
            tokenized_inputs["seg_id"] = seg_ids

        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            with_indices=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.use_segboot:
            prior = type_prior_from_features(train_dataset["labels"], train_dataset["word_start"], label_list)
            model.segboot.log_prior.copy_(torch.tensor(prior).clamp(min=1e-6).log())
            logger.info(f"SegBoot type prior {model.segboot.type_names}: {prior}")
        

    if training_args.do_eval:
        validation_name = "validation" if data_args.eval_on == "dev" else "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[validation_name]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            with_indices=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            with_indices=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )
    

    # Metrics
    metric = load_metric("seqeval")
    eval_ref = None
    if training_args.do_eval:
        raw_eval = datasets[validation_name]
        if data_args.max_val_samples is not None:
            raw_eval = raw_eval.select(range(data_args.max_val_samples))
        eval_ref = {
            "gold": [[label_list[label_to_id[l]] for l in seq] for seq in raw_eval[label_column_name]],
            "orig_word_id": eval_dataset["orig_word_id"],
            "doc_idx": eval_dataset["doc_idx"],
        }

    def compute_metrics(p):
        raw_predictions, labels = p
        logits_np, groups_np, types_np = unpack_predictions(raw_predictions)
        predictions = np.argmax(logits_np, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        out = {"precision": results["overall_precision"], "recall": results["overall_recall"],
               "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
        if eval_ref is not None and predictions.shape[0] == len(eval_ref["doc_idx"]):
            out["entity_f1"] = entity_set_prf(predictions, label_list, eval_ref["orig_word_id"],
                                              eval_ref["doc_idx"], eval_ref["gold"])["entity_f1"]
            if groups_np is not None:
                out["group_entity_f1"] = entity_set_prf_groups(
                    groups_np, types_np, model.segboot.type_names,
                    eval_ref["orig_word_id"], eval_ref["doc_idx"], eval_ref["gold"])["group_entity_f1"]
        return out
    class CustomTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                if data_args.legacy_optimizer:
                    bb = [p for n, p in self.model.named_parameters()
                          if "layoutlmv3" in n and p.requires_grad]
                    new = [p for n, p in self.model.named_parameters()
                           if "layoutlmv3" not in n and p.requires_grad]
                    self.optimizer = torch.optim.AdamW(
                        [{"params": bb, "lr": self.args.learning_rate},
                         {"params": new, "lr": 5e-4}],
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        eps=self.args.adam_epsilon,
                    )  # không truyền weight_decay -> mặc định 0.01, giống bản 0,95
                    return self.optimizer
                NEW_KEYS = ("hierarchical_proj", "line_position_embeddings",
                            "block_position_embeddings", "column_position_embeddings")
                def is_new(n):
                    return (not n.startswith("layoutlmv3.")) or any(k in n for k in NEW_KEYS)
                groups = {}
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    new = is_new(n)
                    decay = p.ndim >= 2          # bias và LayerNorm -> không weight decay
                    groups.setdefault((new, decay), []).append(p)
                param_groups = [
                    {"params": ps,
                     "lr": model_args.new_param_lr if new else self.args.learning_rate,
                     "weight_decay": self.args.weight_decay if decay else 0.0}
                    for (new, decay), ps in groups.items()
                ]
                self.optimizer = torch.optim.AdamW(
                    param_groups, betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon)
            return self.optimizer

    callbacks = []
    if data_args.use_segboot:
        callbacks.append(SegBootEpsCallback(model_args.segboot_eps_start, model_args.segboot_eps_end,
                                            model_args.segboot_eps_decay))

    # Khởi tạo Trainer bằng CustomTrainer vừa tạo thay vì Trainer mặc định
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        raw_predictions, labels, metrics = trainer.predict(test_dataset)
        logits_np, groups_np, types_np = unpack_predictions(raw_predictions)
        predictions = np.argmax(logits_np, axis=2)

        raw_test = datasets["test"]
        if data_args.max_test_samples is not None:
            raw_test = raw_test.select(range(data_args.max_test_samples))
        gold = [[label_list[label_to_id[l]] for l in seq] for seq in raw_test[label_column_name]]
        ent = entity_set_prf(predictions, label_list, test_dataset["orig_word_id"],
                             test_dataset["doc_idx"], gold)
        if groups_np is not None:
            ent.update(entity_set_prf_groups(groups_np, types_np, model.segboot.type_names,
                                             test_dataset["orig_word_id"], test_dataset["doc_idx"], gold))
        trainer.log_metrics("test_entity", ent)
        trainer.save_metrics("test_entity", ent)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # 1. Trích xuất nhãn thực tế (Gold) và chuỗi Tokens gốc
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_tokens = [
            [tokenizer.convert_ids_to_tokens(t) for (t, l) in zip(input_ids, label) if l != -100]
            for input_ids, label in zip(test_dataset["input_ids"], labels)
        ]

        # 2. Phân tích và ghi danh sách token bị lỗi ra file
        output_error_file = os.path.join(training_args.output_dir, "test_errors_analysis.txt")
        if trainer.is_world_process_zero():
            error_count = 0
            with open(output_error_file, "w", encoding="utf-8") as writer:
                writer.write(f"{'Doc_Idx':<10} | {'Token':<25} | {'Gold (True)':<15} | {'Predicted':<15}\n")
                writer.write("-" * 75 + "\n")
                
                for f, (t_tokens, t_preds, t_labels) in enumerate(zip(true_tokens, true_predictions, true_labels)):
                    # Lấy chỉ số tài liệu thật từ dataset
                    real_doc_idx = test_dataset["doc_idx"][f]
                    
                    for token, pred, true_lb in zip(t_tokens, t_preds, t_labels):
                        if pred != true_lb:  # Chỉ lọc các token dự đoán sai
                            writer.write(f"{real_doc_idx:<10} | {token:<25} | {true_lb:<15} | {pred:<15}\n")
                            error_count += 1
                            
                writer.write("-" * 75 + "\n")
                writer.write(f"Total token-level errors: {error_count}\n")
            
            logger.info(f"*** Đã lưu danh sách token dự đoán lỗi tại: {output_error_file} ***")

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()