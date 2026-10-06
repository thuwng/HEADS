#!/bin/bash
# =============================================================================
# Thay cho scripts/run_funsd_base.sh. Cùng một script cho FUNSD và CORD.
#
#   DATASET=funsd PROTOCOL=final SETTING=A MODEL=B0      bash scripts/run_kie.sh   # tái hiện paper (~90.3)
#   DATASET=funsd PROTOCOL=final SETTING=B MODEL=B0      bash scripts/run_kie.sh   # BASELINE ĐÚNG cho B
#   DATASET=funsd PROTOCOL=dev   SETTING=B MODEL=SegBoot TAU=0.5 bash scripts/run_kie.sh   # chọn siêu tham số
#   DATASET=funsd PROTOCOL=final SETTING=B MODEL=SegBoot bash scripts/run_kie.sh   # số báo cáo
#   ... ORACLE=1 (chỉ SegBoot): nhóm gold ở test -> upper bound, KHÔNG báo cáo như kết quả chính
#
# SETTING : A = segment box gold + thứ tự gốc (giao thức LayoutLMv3, chỉ để đối chiếu 90.29)
#           B = word box + thứ tự gốc
#           C = word box + XY-Cut
# METRIC CHÍNH (main_f1):
#   A, B : test_f1  = seqeval trên BIO mức từ, đúng cách tính của LayoutLMv3
#   C    : entity_f1 = khớp (type, tập từ gốc) với gold GỐC. seqeval ở C chấm trên nhãn gold đã
#          retag theo thứ tự XY-Cut (entity bị cắt thành nhiều gold) nên bị thổi phồng -> không dùng.
# Chỉ so SegBoot với B0 CÙNG DATASET + SETTING + PROTOCOL (script tự in delta nếu đã chạy B0).
# =============================================================================
set -e

cd /home/tahuuloc/Documents/tthu/HEADS
export PYTHONPATH="/home/tahuuloc/Documents/tthu/HEADS:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

DATASET=${DATASET:-funsd}
PROTOCOL=${PROTOCOL:-final}
SETTING=${SETTING:-B}
MODEL=${MODEL:-B0}
ORACLE=${ORACLE:-0}
SEEDS=(${SEEDS:-42 123 1993})

case "$SETTING" in
  A) SETTING_FLAGS="--bbox_level segment --apply_xy_cut False" ;;
  B) SETTING_FLAGS="--bbox_level word --apply_xy_cut False" ;;
  C) SETTING_FLAGS="--bbox_level word --apply_xy_cut True" ;;
  *) echo "SETTING phải là A, B hoặc C"; exit 1 ;;
esac

# Siêu tham số fine-tune theo paper LayoutLMv3 (mục 3.3), quy về 1 GPU bằng gradient accumulation.
case "$DATASET" in
  funsd) LR=1e-5; BS=2; GA=8 ;;     # batch 16, lr 1e-5, 1000 step
  cord)  LR=5e-5; BS=2; GA=32 ;;    # batch 64, lr 5e-5, 1000 step
  *) echo "DATASET phải là funsd hoặc cord"; exit 1 ;;
esac

case "$MODEL" in
  B0)      MODEL_FLAGS=""; NEW_LR="--new_param_lr ${LR}" ;;
  SegBoot) MODEL_FLAGS="--use_segboot_v2 --segboot_knn ${KNN:-24} --segboot_tau ${TAU:-0.5} \
                        --segboot_lambda_group ${LG:-1.0} --segboot_lambda_aux ${LA:-0.5} \
                        --segboot_eps_start ${EPS_START:-1.0} --segboot_eps_end ${EPS_END:-0.0} \
                        --segboot_eps_decay ${EPS_DECAY:-0.5}"
           NEW_LR="--new_param_lr ${HEAD_LR:-5e-4}"
           [ "$ORACLE" = "1" ] && MODEL_FLAGS="$MODEL_FLAGS --segboot_eval_oracle True" ;;
  *) echo "MODEL phải là B0 hoặc SegBoot"; exit 1 ;;
esac

if [ "$PROTOCOL" = "dev" ]; then
  PROTO_FLAGS="--eval_on dev --dev_ratio 0.2 --dev_seed ${DEV_SEED:-42} --do_train --do_eval"
elif [ "$PROTOCOL" = "final" ]; then
  PROTO_FLAGS="--eval_on test --do_train --do_eval --do_predict"
else
  echo "PROTOCOL phải là dev hoặc final"; exit 1
fi

TAG="${DATASET}-${PROTOCOL}-${SETTING}-${MODEL}"
[ "$ORACLE" = "1" ] && TAG="${TAG}-ORACLE"
TAG="${TAG}${EXTRA_TAG}"
BASE_OUT_DIR="./logs/${TAG}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT_DIR}-seed${SEED}"
  echo "=== ${TAG} | seed ${SEED} ==="
  # Không warmup, không weight decay, không chọn checkpoint (giữ checkpoint cuối) như script chính thức.
  python examples/run_funsd_cord.py \
    --dataset_name "$DATASET" \
    $PROTO_FLAGS $SETTING_FLAGS $MODEL_FLAGS $NEW_LR \
    --model_name_or_path models/layoutlmv3-base \
    --output_dir "$OUT_DIR" \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
    --dataloader_num_workers 4 --report_to none \
    --seed "$SEED" --legacy_optimizer False \
    --overwrite_output_dir --overwrite_cache
done

export BASE_OUT_DIR DATASET PROTOCOL SETTING MODEL SEEDS_STR="${SEEDS[*]}"
python - <<'PY'
import os, json, numpy as np
base, proto, setting = os.environ["BASE_OUT_DIR"], os.environ["PROTOCOL"], os.environ["SETTING"]
seeds = os.environ["SEEDS_STR"].split()
if proto == "dev":
    pre = "eval_"
    files = {"eval_results.json": None}
else:
    pre = ""
    files = {"test_results.json": None, "test_entity_results.json": None}
main_key = (pre + "entity_f1") if setting == "C" else (pre + "f1" if proto == "dev" else "test_f1")
res = {}
for s in seeds:
    for fn in files:
        p = f"{base}-seed{s}/{fn}"
        if not os.path.exists(p):
            print("[thiếu]", p); continue
        for k, v in json.load(open(p)).items():
            if isinstance(v, (int, float)) and ("f1" in k or "precision" in k or "recall" in k
                                               or "exact_rate" in k or "gate" in k):
                res.setdefault(k, []).append(float(v))
summary = {}
for k, v in sorted(res.items()):
    summary[k] = {"values": v, "mean": float(np.mean(v)),
                  "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
if main_key in summary:
    summary["main_f1"] = dict(summary[main_key], source=main_key)
for k, d in summary.items():
    print(f"{k:34s} {100*d['mean']:6.2f} ± {100*d['std']:4.2f}   (n={len(d['values'])})")
json.dump(summary, open(f"{base}_summary.json", "w"), indent=2)
print("Saved:", f"{base}_summary.json")

# So với B0 cùng dataset/protocol/setting (nếu đã chạy)
b0 = f"./logs/{os.environ['DATASET']}-{proto}-{setting}-B0_summary.json"
if os.environ["MODEL"] != "B0" and os.path.exists(b0) and "main_f1" in summary:
    ref = json.load(open(b0)).get("main_f1")
    if ref:
        d = 100 * (summary["main_f1"]["mean"] - ref["mean"])
        print(f"main_f1 so với B0 ({os.path.basename(b0)}): {d:+.2f} điểm "
              f"[B0 {100*ref['mean']:.2f} ± {100*ref['std']:.2f}]")
PY