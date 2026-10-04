#!/bin/bash
# Cách dùng:
#   SETTING=A MODEL=Seg    bash scripts/run_funsd_base_segment.sh   # tái hiện 0,95
#   SETTING=C MODEL=Latent bash scripts/run_funsd_base_segment.sh   # mô hình mới, thực tế
#   SETTING=C MODEL=B0     bash scripts/run_funsd_base_segment.sh   # baseline cùng điều kiện
#
# SETTING: A = segment box + thứ tự gốc (giao thức LayoutLMv3)
#          B = word box + thứ tự gốc
#          C = word box + XY-Cut (không oracle)
# MODEL:   B0 = LayoutLMv3 thuần | Seg = segment head (bản 0,95) | Latent = mô hình mới
set -e

cd /home/tahuuloc/Documents/tthu/HEADS
export PYTHONPATH="/home/tahuuloc/Documents/tthu/HEADS:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

SETTING=${SETTING:-A}
MODEL=${MODEL:-Seg}
SEEDS=(42 123 1993)

# ---- Setting -----------------------------------------------------------------
case "$SETTING" in
  A) SETTING_FLAGS="--bbox_level segment --apply_xy_cut False"; SEG_SRC="oracle_bbox" ;;
  B) SETTING_FLAGS="--bbox_level word --apply_xy_cut False";    SEG_SRC="line" ;;
  C) SETTING_FLAGS="--bbox_level word --apply_xy_cut True";     SEG_SRC="line" ;;
  *) echo "SETTING phải là A, B hoặc C"; exit 1 ;;
esac

# ---- Mô hình -----------------------------------------------------------------
case "$MODEL" in
  B0)     MODEL_FLAGS="" ;;
  Seg)    MODEL_FLAGS="--use_segment_head --seg_source ${SEG_SRC}" ;;
  Latent) MODEL_FLAGS="--use_latent_segment --lambda_boundary 0.5 \
                       --lds_use_ctx True --lds_use_gate True --lds_use_start_cue True" ;;
  *) echo "MODEL phải là B0, Seg hoặc Latent"; exit 1 ;;
esac

# Thư mục riêng cho từng (setting, model) -> không ghi đè kết quả 0,95 cũ
BASE_OUT_DIR="./logs/funsd-${SETTING}-${MODEL}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
  echo ""
  echo "============================================================"
  echo "FUNSD | SETTING=${SETTING} | MODEL=${MODEL} | SEED=${SEED}"
  echo "============================================================"

  python examples/run_funsd_cord.py \
    --dataset_name funsd \
    --do_train --do_eval --do_predict \
    $SETTING_FLAGS $MODEL_FLAGS \
    --model_name_or_path models/layoutlmv3-base \
    --output_dir "$OUT_DIR" \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 1e-5 --warmup_ratio 0.1 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --dataloader_num_workers 4 --report_to none \
    --seed "$SEED" \
    --overwrite_output_dir --overwrite_cache \
    --use_hierarchical_position_encoding \
    --max_line_position 100 --max_block_position 30 \
    --use_column_encoding True --max_column_position 8 \
    --hipos_impl legacy --legacy_optimizer True
done

echo ""
echo "============================================================"
echo "3-SEED MEAN ± STD | SETTING=${SETTING} | MODEL=${MODEL}"
echo "============================================================"

export BASE_OUT_DIR SETTING
python - <<'PY'
import os, json, numpy as np
seeds = [42, 123, 1993]
base_dir, setting = os.environ["BASE_OUT_DIR"], os.environ["SETTING"]
sources = {
    # seqeval: giống hệt giao thức LayoutLMv3 (dùng làm metric chính ở setting A/B)
    "eval_results.json":        ["eval_f1", "eval_precision", "eval_recall"],
    # entity_set_prf: không phụ thuộc thứ tự đọc (metric chính ở setting C)
    "test_entity_results.json": ["entity_f1", "entity_precision", "entity_recall"],
}
results = {m: [] for ms in sources.values() for m in ms}
for seed in seeds:
    print(f"\nSeed {seed}:")
    for fname, metrics in sources.items():
        path = f"{base_dir}-seed{seed}/{fname}"
        if not os.path.exists(path):
            print("  [WARNING] Missing:", path); continue
        data = json.load(open(path))
        for m in metrics:
            if m in data:
                results[m].append(float(data[m]))
                print(f"  {m:18s} = {data[m]:.6f}")

main = "eval_f1" if setting in ("A", "B") else "entity_f1"
print("\n" + "=" * 70)
print(f"FINAL RESULT (metric chính cho setting {setting}: {main})")
print("=" * 70)
summary = {}
for m, vals in results.items():
    if not vals:
        continue
    mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    flag = "  <== chính" if m == main else ""
    print(f"{m:18s}: {mean:.4f} ± {std:.4f}{flag}")
    summary[m] = {"values": vals, "mean": mean, "std": std}
out_file = f"{base_dir}_3seed_summary.json"
json.dump(summary, open(out_file, "w"), indent=2)
print("=" * 70)
print("Saved:", out_file)
PY