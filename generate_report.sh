CKPT_NAME='finetune-visualglm-6b-freeze-0,14'
PREFIX='COV-CTR'
CKPT_PATH="/home/qianq/mycodes/VisualGLM-6B/checkpoints/$PREFIX/$CKPT_NAME"
REPORT_SAVE_PATH="/home/qianq/mycodes/VisualGLM-6B/reports/$PREFIX/$CKPT_NAME.jsonl"

CUDA_VISIBLE_DEVICES=2 python generate_report.py \
    --ckpt_path ${CKPT_PATH} \
    --report_save_path ${REPORT_SAVE_PATH}