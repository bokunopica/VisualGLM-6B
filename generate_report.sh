CKPT_NAME='finetune-visualglm-eva-qformer'
CKPT_PATH="/home/qianq/mycodes/VisualGLM-6B/checkpoints/$CKPT_NAME"
REPORT_SAVE_PATH="/home/qianq/mycodes/VisualGLM-6B/reports/$CKPT_NAME.jsonl"

CUDA_VISIBLE_DEVICES=2 python generate_report.py \
    --ckpt_path ${CKPT_PATH} \
    --report_save_path ${REPORT_SAVE_PATH}