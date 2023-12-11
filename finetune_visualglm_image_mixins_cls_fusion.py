import os
import torch
import argparse
import json
from PIL import Image
from tqdm import trange

from torch.utils.data import Dataset
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model.blip2 import BlipImageEvalProcessor
from model.visualglm import FineTuneVisualGLMModel
from transformers import AutoTokenizer


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
            is_covid = item['is_covid']
            if is_covid:
                disease_prompt = "该患者患有新冠肺炎。"
            else:
                disease_prompt = "该患者未患新冠肺炎。"
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + ""+ "\n答：", add_special_tokens=False
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


def create_dataset_function(path, args):
    # tokenizer = get_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
    image_processor = BlipImageEvalProcessor(224)
    dataset = XrayDataset(path, image_processor, tokenizer, args)
    # dataset = MimicXrayDataset(path, image_processor, tokenizer, args)
    return dataset


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max_source_length", type=int)
    py_parser.add_argument("--max_target_length", type=int)
    py_parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument("--source_prefix", type=str, default="")
    py_parser.add_argument("--use_classification_info", action="store_true")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.device = "cpu"

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

    if torch.cuda.is_available():
        model = model.to("cuda")
    # tokenizer = get_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
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
        }
        return ret

    training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step,
        create_dataset_function=create_dataset_function,
        collate_fn=data_collator,
    )
