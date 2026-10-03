#!/bin/bash
set -e


cd /home/tahuuloc/Documents/tthu/HEADS
export PYTHONPATH="/home/tahuuloc/Documents/tthu/HEADS:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true 

SEEDS=(42 123 1993)
BASE_OUT_DIR="./logs/funsd-base-segment-final"

# Setting 1 (oracle): tái hiện model cũ. Setting 2 xem ghi chú bên dưới.
# SETTING_FLAGS="--bbox_level segment --seg_source oracle_bbox --apply_xy_cut False"
# Setting 2
SETTING_FLAGS="--bbox_level word --apply_xy_cut True"
for SEED in "${SEEDS[@]}"
do
    OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
    
    echo ""
    echo "============================================================"
    echo "RUNNING FUNSD BASE (SEGMENT HEAD + HIPOS) - SEED = ${SEED}"
    echo "============================================================"
    

    python examples/run_funsd_cord.py \
      --dataset_name funsd \
      --do_train --do_eval --do_predict \
      --use_latent_segment --lambda_boundary 0.5 \
      --lds_use_ctx True --lds_use_gate True --lds_use_start_cue True \
      --use_segment_head False \
      $SETTING_FLAGS \
      --model_name_or_path models/layoutlmv3-base \
      --output_dir "$OUT_DIR" \
      --visual_embed 1 --input_size 224 \
      --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --dataloader_num_workers 4 \
      --report_to none \
      --seed "$SEED" \
      --overwrite_output_dir --overwrite_cache \
      --use_hierarchical_position_encoding False \
      --use_column_encoding False
done

echo ""
echo "============================================================"
echo "CALCULATING 3-SEED MEAN ± STD (FUNSD BASE)"
echo "============================================================"

export BASE_OUT_DIR
python - <<'PY'
import os, json, numpy as np
seeds = [42, 123, 1993]
metrics = ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_loss"]
results = {m: [] for m in metrics}
results["test_entity_f1"] = []
base_dir = os.environ['BASE_OUT_DIR']

for seed in seeds:
    print(f"\nSeed {seed}:")
    
    # 1. Đọc Seqeval F1 truyền thống
    path = f"{base_dir}-seed{seed}/test_results.json"
    if not os.path.exists(path):
        print("  [WARNING] Missing:", path)
    else:
        with open(path, "r") as f:
            data = json.load(f)
        for metric in metrics:
            if metric in data:
                val = float(data[metric])
                if metric != "test_loss" and val <= 1.0: val *= 100.0
                results[metric].append(val)
                print(f"  {metric:18s} = {val:.4f}")
                
    # 2. Đọc Entity-Set F1 (SỐ DÙNG CHO PAPER)
    path_ent = f"{base_dir}-seed{seed}/test_entity_results.json"
    if os.path.exists(path_ent):
        with open(path_ent, "r") as f:
            data_ent = json.load(f)
        # Bắt đúng tên key do trainer sinh ra
        ent_key = "test_entity_entity_f1" if "test_entity_entity_f1" in data_ent else "entity_f1"
        if ent_key in data_ent:
            val = float(data_ent[ent_key])
            if val <= 1.0: val *= 100.0
            results["test_entity_f1"].append(val)
            print(f"  test_entity_f1     = {val:.4f}  <-- DÙNG SỐ NÀY CHO PAPER")

print("\n" + "=" * 70)
print("FINAL RESULT: MEAN ± STD")
print("=" * 70)
summary = {}
for metric in metrics + ["test_entity_f1"]:
    values = results[metric]
    if not values:
        continue
    mean, std = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0
    marker = " (MAIN METRIC)" if metric == "test_entity_f1" else ""
    print(f"{metric:18s}: {mean:.4f} ± {std:.4f}{marker}")
    summary[metric] = {"values": values, "mean": float(mean), "std": float(std)}

out_file = f"{base_dir}_3seed_summary.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)
print("=" * 70)
print("Saved:", out_file)
PY