#!/bin/bash
# =============================================================================
# Cách dùng
#   PROTOCOL=final SETTING=A MODEL=B0      bash scripts/run_funsd_segboot.sh  # (1) tái hiện LayoutLMv3 (~90.3)
#   PROTOCOL=dev   SETTING=B MODEL=SegBoot bash scripts/run_funsd_segboot.sh  # (2) tinh chỉnh trên DEV
#   PROTOCOL=final SETTING=B MODEL=SegBoot bash scripts/run_funsd_segboot.sh  # (3) số báo cáo cuối
#   PROTOCOL=final SETTING=B MODEL=SegBoot ORACLE=1 bash ...                   # (4) upper bound (phân tích)
#
# PROTOCOL: dev   = train trên 80% train, eval trên 20% dev, KHÔNG chạy test
#           final = train trên toàn bộ train, 1000 step cố định, checkpoint CUỐI, báo cáo test
#                   (đúng như script chính thức của LayoutLMv3: không chọn checkpoint)
# SETTING : A = segment box + thứ tự gốc  (giao thức LayoutLMv3 – chỉ để so số 90.29)
#           B = word box + thứ tự gốc      (bỏ oracle segment – setting CHÍNH của paper)
#           C = word box + XY-Cut          (bỏ oracle segment + thứ tự đọc thực tế)
# MODEL   : B0 = LayoutLMv3 thuần | Seg = segment head (bản 0,95) | Latent | SegBoot
# HIPOS=1 : bật hierarchical/column position (chỉ dùng như một ablation riêng)
# =============================================================================
set -e

cd /home/tahuuloc/Documents/tthu/HEADS
export PYTHONPATH="/home/tahuuloc/Documents/tthu/HEADS:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

PROTOCOL=${PROTOCOL:-final}
SETTING=${SETTING:-A}
MODEL=${MODEL:-B0}
HIPOS=${HIPOS:-0}
ORACLE=${ORACLE:-0}
SEEDS=(${SEEDS:-42 123 1993})

case "$SETTING" in
  A) SETTING_FLAGS="--bbox_level segment --apply_xy_cut False"; SEG_SRC="oracle_bbox" ;;
  B) SETTING_FLAGS="--bbox_level word --apply_xy_cut False";    SEG_SRC="line" ;;
  C) SETTING_FLAGS="--bbox_level word --apply_xy_cut True";     SEG_SRC="line" ;;
  *) echo "SETTING phải là A, B hoặc C"; exit 1 ;;
esac

# Mặc định: cùng một learning rate cho MỌI tham số, giống LayoutLMv3 chính thức.
NEW_LR="--new_param_lr 1e-5"
case "$MODEL" in
  B0)      MODEL_FLAGS="" ;;
  Seg)     MODEL_FLAGS="--use_segment_head --seg_source ${SEG_SRC}" ;;
  Latent)  MODEL_FLAGS="--use_latent_segment --lambda_boundary 0.5" ; NEW_LR="--new_param_lr ${HEAD_LR:-5e-4}" ;;
  SegBoot) MODEL_FLAGS="--use_segboot --segboot_knn ${KNN:-24} --segboot_tau 0.5 \
                        --segboot_lambda_group ${LG:-1.0} --segboot_logit_adj ${LA:-1.0} \
                        --segboot_eps_start 1.0 --segboot_eps_end ${EPS_END:-0.0} --segboot_eps_decay ${EPS_DECAY:-0.6} \
                        --segboot_final_groups ${FINAL_GROUPS:-pass2} --order_aug_prob ${ORDER_AUG:-0.8}"
           NEW_LR="--new_param_lr ${HEAD_LR:-5e-4}"
           [ "$ORACLE" = "1" ] && MODEL_FLAGS="$MODEL_FLAGS --segboot_eval_oracle True" ;;
  *) echo "MODEL phải là B0, Seg, Latent hoặc SegBoot"; exit 1 ;;
esac
# Lưu ý: SegBoot ở setting A không có ý nghĩa (box đầu vào đã là oracle) – chỉ chạy như sanity check.

if [ "$HIPOS" = "1" ]; then
  HIPOS_FLAGS="--use_hierarchical_position_encoding --max_line_position 100 --max_block_position 30 \
               --use_column_encoding True --max_column_position 8 --hipos_impl legacy"
else
  HIPOS_FLAGS=""
fi

if [ "$PROTOCOL" = "dev" ]; then
  PROTO_FLAGS="--eval_on dev --dev_ratio 0.2 --dev_seed ${DEV_SEED:-42} --do_train --do_eval"
elif [ "$PROTOCOL" = "final" ]; then
  PROTO_FLAGS="--eval_on test --do_train --do_eval --do_predict"
else
  echo "PROTOCOL phải là dev hoặc final"; exit 1
fi

TAG="${PROTOCOL}-${SETTING}-${MODEL}-hipos${HIPOS}"
[ "$ORACLE" = "1" ] && TAG="${TAG}-ORACLE"

if [ "$MODEL" = "SegBoot" ]; then
  TAG="${TAG}-aug${ORDER_AUG:-0.0}"
fi

TAG="${TAG}${EXTRA_TAG}"
BASE_OUT_DIR="./logs/funsd-${TAG}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
  echo "=== FUNSD | ${TAG} | seed ${SEED} ==="
  # Siêu tham số chung = README LayoutLMv3 (8 GPU x bs2) quy về 1 GPU: bs2 x GA8 = 16.
  # KHÔNG warmup, KHÔNG weight decay, KHÔNG load_best_model_at_end (giữ checkpoint cuối).
  python examples/run_funsd_cord.py \
    --dataset_name funsd \
    $PROTO_FLAGS $SETTING_FLAGS $MODEL_FLAGS $HIPOS_FLAGS $NEW_LR \
    --model_name_or_path models/layoutlmv3-base \
    --output_dir "$OUT_DIR" \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --dataloader_num_workers 4 --report_to none \
    --seed "$SEED" --legacy_optimizer False \
    --overwrite_output_dir --overwrite_cache
done

export BASE_OUT_DIR PROTOCOL SEEDS_STR="${SEEDS[*]}"
python - <<'PY'
import os, json, numpy as np
base, proto = os.environ["BASE_OUT_DIR"], os.environ["PROTOCOL"]
seeds = os.environ["SEEDS_STR"].split()
if proto == "dev":
    sources = {"eval_results.json": ["eval_f1", "eval_entity_f1", "eval_group_entity_f1"]}
else:
    sources = {
        "test_results.json": ["test_f1", "test_precision", "test_recall"],               # seqeval (giao thức LayoutLMv3)
        "test_entity_results.json": ["entity_f1", "entity_f1_HEADER", "entity_f1_QUESTION", "entity_f1_ANSWER",
                                     "group_entity_f1", "group_entity_f1_HEADER",
                                     "group_entity_f1_QUESTION", "group_entity_f1_ANSWER"],
    }
res = {}
for s in seeds:
    for fn, keys in sources.items():
        p = f"{base}-seed{s}/{fn}"
        if not os.path.exists(p):
            print("[thiếu]", p); continue
        d = json.load(open(p))
        for k in keys:
            if k in d:
                res.setdefault(k, []).append(float(d[k]))
summary = {}
for k, v in res.items():
    m, sd = float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    summary[k] = {"values": v, "mean": m, "std": sd}
    print(f"{k:28s} {100*m:6.2f} ± {100*sd:4.2f}   (n={len(v)})")
json.dump(summary, open(f"{base}_summary.json", "w"), indent=2)
print("Saved:", f"{base}_summary.json")
PY
