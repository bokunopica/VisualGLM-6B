#! /bin/bash
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=4
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b"
MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256"

OPTIONS_DEVICE="CUDA_VISIBLE_DEVICES=0,2"
# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_P2P_DISABLE=1"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

# train_data="./fewshot-data/dataset.json"
# eval_data="./fewshot-data/dataset.json"
# train_data="/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-train-prompt.json"
# eval_data="/home/qianq/data/OpenI-zh-resize-384/images/openi-zh-test-prompt.json"
train_data="/home/qianq/data/COV-CTR/train.json"
eval_data="/home/qianq/data/COV-CTR/eval.json"



gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 3000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 1000 \
       --eval-interval 50 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 4 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size 4 \
       --skip-init \
       --fp16 \
       --use_freeze \
       --unfreeze_layers 0,14
"

              

run_cmd="${OPTIONS_DEVICE} ${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_visualglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
