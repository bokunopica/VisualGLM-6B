CKPT_NAME='finetune-visualglm-6b-qformer'
PREFIX='COV-CTR-30k'
SEED=1919810
CKPT_PATH="checkpoints/$PREFIX/$CKPT_NAME"
REPORT_SAVE_PATH="reports/$PREFIX-seed$SEED/$CKPT_NAME.jsonl"

CUDA_VISIBLE_DEVICES=3 python generate_report.py \
    --ckpt_path ${CKPT_PATH} \
    --report_save_path ${REPORT_SAVE_PATH} \
    --seed ${SEED}
    # --use_covid_tag