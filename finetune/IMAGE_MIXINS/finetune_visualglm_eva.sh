#! /bin/bash
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b-eva"
MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256"

OPTIONS_DEVICE="CUDA_VISIBLE_DEVICES=0,2"
# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

# train_data="./fewshot-data/dataset.json"
# eval_data="./fewshot-data/dataset.json"
train_data="/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-prompt.json"
eval_data="/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-prompt.json"
# train_data="/home/qianq/data/balance_mimic_pneumonia/train_metadata_final.json"
# eval_data="/home/qianq/data/balance_mimic_pneumonia/train_metadata_final.json"


gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 10000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 2000 \
       --eval-interval 10 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size 4 \
       --skip-init \
       --fp16 \
       --train_qformer
"

              

run_cmd="${OPTIONS_DEVICE} ${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_visualglm_image_mixins.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
