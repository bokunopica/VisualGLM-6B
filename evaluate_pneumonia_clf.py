import torch
import argparse
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model.visualglm import FineTuneVisualGLMModel
from transformers import AutoTokenizer
from dataset import CovCTRDataset

from model.vit_classifier import PneumoniaClassifier
from model.blip2 import BlipImageEvalProcessor
from tqdm import trange
import torch.nn.functional as F


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ["is_covid_one_hot"]
    datatype = torch.int64
    # Broadcast data.
    timers("data loader").start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers("data loader").stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    data_i = mpu.broadcast_data(["image"], data, torch.float32)
    # Unpack.
    # tokens = data_b["input_ids"].long()
    # labels = data_b["labels"].long()
    is_covid_one_hot = data_b["is_covid_one_hot"].long()
    # is_covid = torch.nn.functional.one_hot(is_covid, num_classes=2)
    img = data_i["image"]
    if args.fp16:
        img = img.half()
    return img, is_covid_one_hot


from torch.nn import CrossEntropyLoss


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers("batch generator").start()
    image, is_covid_one_hot = get_batch(data_iterator, args, timers)

    timers("batch generator").stop()
    logits = model(image=image)
    # dtype = logits.dtype
    # lm_logits = logits.to(torch.float32)

    # # Shift so that tokens < n predict n
    # shift_logits = lm_logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()

    # # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # lm_logits = lm_logits.to(dtype)
    # loss = loss.to(dtype)
    dtype = logits.dtype
    cmp_logits = logits.to(torch.float32)
    is_covid_one_hot = is_covid_one_hot.to(torch.float32)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(cmp_logits, is_covid_one_hot)

    cmp_logits = cmp_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {"loss": loss}


def create_dataset_function(path, args):
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/qianq/model/chatglm-6b", trust_remote_code=True
    )
    image_processor = BlipImageEvalProcessor(224)
    dataset = CovCTRDataset(path, image_processor, tokenizer, args)
    return dataset


if __name__ == "__main__":
    args = argparse.Namespace(
        num_layers=6,
        hidden_size=1024,
        num_attention_heads=16,
        vocab_size=100,
        max_sequence_length=512,
        layernorm_order="pre",
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        model_parallel_size=1,
        skip_init=True,
        use_gpu_initialization=False,
        num_multi_query_heads=0,
        layernorm_epsilon=1e-05,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.0,
        make_vocab_size_divisible_by=128,
        experiment_name="finetune-eva-vit-classifier",
        train_iters=3000,
        batch_size=64,
        lr=0.0001,
        mode="finetune",
        seed=1234,
        zero_stage=1,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        checkpoint_skip_layers=0,
        fp16=True,
        bf16=False,
        gradient_accumulation_steps=1,
        epochs=None,
        log_interval=50,
        summary_dir="",
        save_args=False,
        lr_decay_iters=None,
        lr_decay_style="cosine",
        lr_decay_ratio=0.1,
        warmup=0.02,
        weight_decay=0.01,
        save="./checkpoints",
        load=None,
        save_interval=3000,
        no_save_rng=False,
        no_load_rng=False,
        resume_dataloader=True,
        distributed_backend="nccl",
        local_rank=0,
        exit_interval=None,
        eval_batch_size=64,
        eval_iters=10,
        eval_interval=100,
        strict_eval=False,
        train_data=["/home/qianq/data/COV-CTR/train.json"],
        train_data_weights=None,
        iterable_dataset=False,
        valid_data=["/home/qianq/data/COV-CTR/eval.json"],
        test_data=None,
        split="1",
        num_workers=1,
        block_size=10000,
        prefetch_factor=4,
        tokenizer_type="fake",
        temperature=1.0,
        top_p=0.0,
        top_k=0,
        num_beams=1,
        length_penalty=0.0,
        no_repeat_ngram_size=0,
        min_tgt_length=0,
        out_seq_length=256,
        input_source="interactive",
        output_path="./samples",
        with_id=False,
        max_inference_batch_size=12,
        device="cpu",
        deepspeed=True,
        deepspeed_config={
            "train_micro_batch_size_per_gpu": 64,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 0.1,
            "zero_optimization": {
                "stage": 1,
                "cpu_offload": False,
                "contiguous_gradients": False,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 40000000.0,
                "allgather_bucket_size": 100000000.0,
                "load_from_fp32_weights": False,
            },
            "zero_allow_untested_optimizer": True,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 400,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {"enabled": False},
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.0001,
                    "betas": [0.9, 0.95],
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                },
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "contiguous_memory_optimization": False,
            },
            "wall_clock_breakdown": False,
        },
        deepscale=False,
        deepscale_config=None,
        deepspeed_mpi=False,
        cuda=True,
        rank=0,
        world_size=1,
        deepspeed_activation_checkpointing=True,
        master_ip="127.0.0.1",
        master_port="16666",
        max_source_length=64,
        max_target_length=256,
        ignore_pad_token_for_loss=True,
        source_prefix="",
        pre_seq_len=8,
        lora_rank=10,
        use_ptuning=False,
        use_lora=False,
        use_qlora=False,
        layer_range=None,
        use_adapter=False,
        adapter_hidden=128,
        adapter_num_layers=28,
        use_freeze=False,
        unfreeze_layers="",
        train_qformer=True,
        train_vit_transformer="",
        cls_fusion=False,
        image_length=32,
        eva_args={},
        qformer_args={},
        bos_token_id=None,
        mask_token_id=None,
        gmask_token_id=None,
        pad_token_id=None,
    )
    args.device = "cpu"

    model = PneumoniaClassifier(
        eva_args={
            "num_layers": 39,
            "hidden_size": 1408,
            "num_attention_heads": 16,
            "vocab_size": 1,
            "layernorm_order": "pre",
            "model_parallel_size": 1,
            "max_sequence_length": 257,
            "inner_hidden_size": 6144,
            "use_final_layernorm": False,
            "layernorm_epsilon": 1e-06,
            "image_size": [224, 224],
            "pre_len": 1,
            "post_len": 0,
            "in_channels": 3,
            "num_classes": 0,
            "patch_size": 14,
        }
    )
    # vit冻结
    model.vit.requires_grad_(False)

    if torch.cuda.is_available():
        model = model.to("cuda")


    tokenizer = AutoTokenizer.from_pretrained(
        "/home/qianq/model/chatglm-6b", trust_remote_code=True
    )

    path = '/home/qianq/data/COV-CTR/eval.json'
    image_processor = BlipImageEvalProcessor(224)
    dataset = CovCTRDataset(path, image_processor, tokenizer, args)

    for i in range(len(dataset)):
        dataset[i]

    print(len(dataset))
