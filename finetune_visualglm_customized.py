import os
import torch
import argparse
import json
from PIL import Image
from tqdm import trange

from torch.utils.data import Dataset, DataLoader
from sat import mpu, get_args, get_tokenizer
from transformers import AutoTokenizer
from sat.training.deepspeed_training import training_main
from model.blip2 import BlipImageEvalProcessor
from model.visualglm import FineTuneVisualGLMModel


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ["input_ids", "labels"]
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
    tokens = data_b["input_ids"].long()
    labels = data_b["labels"].long()
    img = data_i["image"]
    if args.fp16:
        img = img.half()
    return tokens, labels, img, data["pre_image"]


from torch.nn import CrossEntropyLoss


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers("batch generator").start()
    tokens, labels, image, pre_image = get_batch(data_iterator, args, timers)
    timers("batch generator").stop()

    # TODO get classification loss
    # classification_loss_fct = CrossEntropyLoss()
    logits = model(input_ids=tokens, image=image, pre_image=pre_image)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {"loss": loss}


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class FewShotDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.images = []
        self.input_ids = []
        self.labels = []
        for item in data:
            image = processor(Image.open(item["img"]).convert("RGB"))
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
        }


@singleton
class XrayDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_path = "/".join(path.split("/")[:-1])
        self.images = []
        self.input_ids = []
        self.labels = []
        for i in trange(len(data)):
            item = data[i]
            img_filename = item["img"].split("/")[-1]
            image = processor(
                Image.open(f"{prefix_path}/{img_filename}").convert("RGB")
            )
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
        }


@singleton
class XrayDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_path = "/".join(path.split("/")[:-1])
        self.images = []
        self.input_ids = []
        self.labels = []
        # self.covid_labels = []
        for i in trange(len(data)):
            item = data[i]
            img_filename = item["img"].split("/")[-1]
            image = processor(
                Image.open(f"{prefix_path}/{img_filename}").convert("RGB")
            )
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            # classification tag
            is_covid = item["is_covid"]
            if is_covid:
                disease_prompt = "该患者患有新冠肺炎。"
            else:
                disease_prompt = "该患者未患新冠肺炎。"
            input2 = tokenizer.encode(
                "</img>" + disease_prompt + "问：" + item["prompt"] + "" + "\n答：",
                add_special_tokens=False,
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)

            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
            # "is_covid": self.covid_labels[idx],
        }


@singleton
class MimicXrayDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_path = "/".join(path.split("/")[:-1])
        self.images = []
        self.input_ids = []
        self.labels = []
        for i in trange(len(data)):
            item = data[i]
            img_filename = item["file_name"]
            image = processor(
                Image.open(f"{prefix_path}/{img_filename}").convert("RGB")
            )
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
        }


def create_dataset_function(path, args, tokenizer=None):
    # tokenizer = get_tokenizer(args)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/qianq/model/chatglm-6b", trust_remote_code=True
        )
    image_processor = BlipImageEvalProcessor(224)
    dataset = XrayDataset(path, image_processor, tokenizer, args)
    # dataset = MimicXrayDataset(path, image_processor, tokenizer, args)
    return dataset


def train_custom(
    args,
    train_dataset,
    eval_dataset,
    collate_fn,
    model_cls=None,
    forward_step_function=None,
):
    print("-------train_custom-------")
    """
    dataset getitem
    {
        "image": self.images[idx],
        "input_ids": self.input_ids[idx],
        "labels": self.labels[idx],
        "pre_image": self.pre_image,
    }
    """
    print(args)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
    )
    model = model_cls
    for _ in train_dataloader:
        print(type(_))
        print(list(_.keys()))
        print(len(_["image"]))  # batch size
        print(_["is_covid"])
        print(len(_["is_covid"]))
        tokens = _["input_ids"]
        image = _["image"]
        pre_image = _["pre_image"]
        model(input_ids=tokens, image=image, pre_image=pre_image)[0]
        break
    print("-------train_custom-------")


def main():
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max_source_length", type=int)
    py_parser.add_argument("--max_target_length", type=int)
    py_parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument("--source_prefix", type=str, default="")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.device = "cpu"
    args = argparse.Namespace(
        use_classification_info=True,
        model_class="VisualGLMModel",
        tokenizer_type="THUDM/chatglm-6b",
        num_layers=28,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=130528,
        layernorm_order="post",
        model_parallel_size=1,
        max_sequence_length=2048,
        image_length=32,
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
        },
        qformer_args={
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "vocab_size": 32,
            "layernorm_order": "post",
            "model_parallel_size": 1,
            "max_sequence_length": 0,
            "is_decoder": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "cross_attn_hidden_size": 1408,
            "layernorm_epsilon": 1e-12,
        },
        bos_token_id=130004,
        mask_token_id=130000,
        gmask_token_id=130001,
        pad_token_id=3,
        image_size=[224, 224],
        pre_len=1,
        post_len=0,
        in_channels=3,
        patch_size=14,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        skip_init=True,
        use_gpu_initialization=False,
        num_multi_query_heads=0,
        layernorm_epsilon=1e-05,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.0,
        make_vocab_size_divisible_by=128,
        experiment_name="finetune-visualglm-6b-eva",
        train_iters=3000,
        batch_size=4,
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
        save_interval=1000,
        no_save_rng=False,
        no_load_rng=False,
        resume_dataloader=True,
        distributed_backend="nccl",
        local_rank=0,
        exit_interval=None,
        eval_batch_size=8,
        eval_iters=10,
        eval_interval=10,
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
            "train_micro_batch_size_per_gpu": 4,
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
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/qianq/model/chatglm-6b", trust_remote_code=True
    )

    # datasets
    # train_dataset=create_dataset_function(args.train_data[0], args, tokenizer)
    # eval_dataset=create_dataset_function(args.valid_data[0], args, tokenizer)

    # model
    model_type = "visualglm-6b"
    model, args = FineTuneVisualGLMModel.from_pretrained(
        model_type, args, build_only=True
    )
    for sub_model_name in model.mixins:
        print(sub_model_name)
        if sub_model_name in ["adapter", "ptuning", "lora"]:
            continue
        model.mixins[sub_model_name].load_state_dict(
            torch.load(
                f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/{sub_model_name}.ckpt"
            )
        )
    model.transformer.load_state_dict(
        torch.load(
            f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/chatglm-6b.ckpt"
        )
    )

    ####### freeze the model and unfreeze the image mixins #######
    # [_.requires_grad_(False) for _ in model.parameters()]
    # [_.requires_grad_(True) for _ in model.mixins.eva.parameters()]
    # for name, param in model.mixins.eva.state_dict().items():
    #     param.requires_grad_(True)
    #     print(name, param.requires_grad)

    # for name, param in model.mixins.eva.state_dict().items():
    #     print(name, param.requires_grad)
    # # model.mixins.eva.requires_grad_(True)
    # # model.eval()
    # # model.mixins.eva.parameters().requires_grad_(True)
    ##############################################################

    # if torch.cuda.is_available():
    #     model = model.to("cuda")

    # tokenizer = get_tokenizer(args)
    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    def data_collator(examples):
        for example in examples:
            example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
            example["labels"] = torch.tensor(example["labels"], dtype=torch.long)
        ret = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "labels": torch.stack([example["labels"] for example in examples]),
            "image": torch.stack([example["image"] for example in examples]),
            "pre_image": example["pre_image"],
            "is_covid": example["is_covid"],
        }
        return ret

    training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step,
        create_dataset_function=create_dataset_function,
        collate_fn=data_collator,
    )

    # train_custom(
    #     args,
    #     model_cls=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     collate_fn=data_collator,
    #     forward_step_function=forward_step,
    # )


def test():
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max_source_length", type=int)
    py_parser.add_argument("--max_target_length", type=int)
    py_parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument("--source_prefix", type=str, default="")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    # args = argparse.Namespace(**vars(args), **vars(known))
    # args.device = "cpu"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/qianq/model/chatglm-6b", trust_remote_code=True
    )

    # datasets
    train_dataset = create_dataset_function(args.train_data[0], args, tokenizer)
    eval_dataset = create_dataset_function(args.valid_data[0], args, tokenizer)

    def data_collator(examples):
        for example in examples:
            example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
            example["labels"] = torch.tensor(example["labels"], dtype=torch.long)
        ret = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "labels": torch.stack([example["labels"] for example in examples]),
            "image": torch.stack([example["image"] for example in examples]),
            "pre_image": example["pre_image"],
            "is_covid": example["is_covid"],
        }
        return ret

    model_type = "visualglm-6b"
    model, args = FineTuneVisualGLMModel.from_pretrained(
        model_type, args, build_only=True
    )
    for sub_model_name in model.mixins:
        print(sub_model_name)
        if sub_model_name in ["adapter", "ptuning", "lora"]:
            continue
        model.mixins[sub_model_name].load_state_dict(
            torch.load(
                f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/{sub_model_name}.ckpt"
            )
        )
    model.transformer.load_state_dict(
        torch.load(
            f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/chatglm-6b.ckpt"
        )
    )

    # train_custom(
    #     args,
    #     model_cls=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     collate_fn=data_collator,
    #     forward_step_function=forward_step,
    # )
    training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step,
        create_dataset_function=create_dataset_function,
        collate_fn=data_collator,
    )


if __name__ == "__main__":
    main()
    # test()
    # test()
