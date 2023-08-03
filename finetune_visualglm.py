import os
import torch
import argparse
import json


from PIL import Image
from tqdm import trange
from torch import nn
from torch.utils.data import Dataset
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model import VisualGLMModel
from model.blip2 import BlipImageEvalProcessor
from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune import AdapterMixin
from sat.model.base_model import non_conflict


class AdjustAdapterMixin(AdapterMixin):
    @non_conflict
    def layer_forward(self, hidden_states, mask, old_impl, *args, **kw_args):
        """
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """
        layer = self.transformer.layers[kw_args["layer_id"]]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        attention_output = attention_output + self.ff2[kw_args["layer_id"]](
            nn.functional.gelu(self.ff1[kw_args["layer_id"]](attention_output))
        )

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)
        mlp_output = mlp_output + self.ff4[kw_args["layer_id"]](
            nn.functional.gelu(self.ff3[kw_args["layer_id"]](mlp_output))
        )

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output


class FineTuneVisualGLMModel(VisualGLMModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(
            args, transformer=transformer, parallel_output=parallel_output, **kw_args
        )
        if args.use_ptuning:
            self.add_mixin(
                "ptuning",
                PTuningV2Mixin(
                    args.num_layers,
                    args.hidden_size // args.num_attention_heads,
                    args.num_attention_heads,
                    args.pre_seq_len,
                ),
            )
        if args.use_lora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                ),
                reinit=True,
            )
            # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
        elif args.use_qlora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                    qlora=True,
                ),
                reinit=True,
            )
        elif args.use_adapter:
            # adapter finetune
            self.add_mixin(
                "adapter",
                AdjustAdapterMixin(
                    num_layers=args.num_layers, # 28-transformer一致
                    hidden_size=args.hidden_size, # 4096
                    adapter_hidden=args.adapter_hidden, # specified in .sh
                ),
            )
            pass
        self.args = args

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group(
            "VisualGLM-finetune", "VisualGLM finetune Configurations"
        )
        group.add_argument("--pre_seq_len", type=int, default=8)
        group.add_argument("--lora_rank", type=int, default=10)
        group.add_argument("--use_ptuning", action="store_true")
        group.add_argument("--use_lora", action="store_true")
        group.add_argument("--use_qlora", action="store_true")
        group.add_argument("--layer_range", nargs="+", type=int, default=None)
        group.add_argument("--use_adapter", action="store_true")
        group.add_argument("--adapter_hidden", type=int, default=128)
        group.add_argument("--adapter_num_layers", type=int, default=28)
        group.add_argument("--use_freeze", action="store_true")
        group.add_argument("--unfreeze_layers", type=str, default="")
        return super().add_model_specific_args(parser)

    def disable_untrainable_params(self):
        enable = []
        if self.args.use_ptuning:
            enable.extend(["ptuning"])
        if self.args.use_lora or self.args.use_qlora:
            enable.extend(["matrix_A", "matrix_B"])
        if self.args.use_freeze:
            unfreeze_layers = args.unfreeze_layers.split(',')
        else:
            unfreeze_layers = []
        print('------------unfreeze_layer--------------')
        for n, p in self.named_parameters():
            flag = False
            # adapter unfreeze
            if self.args.use_adapter and n.startswith("mixins.adapter"):
                flag = True
            elif self.args.use_freeze:
                for unfreeze_layer in unfreeze_layers:
                    if n.startswith(f"transformer.layers.{unfreeze_layer}."):
                        flag = True
                        break
            else:
                for e in enable:
                    if e.lower() in n.lower():
                        flag = True
                        break
            if not flag:
                p.requires_grad_(False)
            else:
                print(n)
        print('------------unfreeze_layer--------------')



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

def create_dataset_function(path, args):
    tokenizer = get_tokenizer(args)
    image_processor = BlipImageEvalProcessor(224)
    dataset = XrayDataset(path, image_processor, tokenizer, args)
    return dataset


if __name__ == "__main__":
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

    model_type = "visualglm-6b"
    model, args = FineTuneVisualGLMModel.from_pretrained(
        model_type, args, build_only=True
    )
    for sub_model_name in model.mixins:
        print(sub_model_name)
        if sub_model_name == "adapter":
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

    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = get_tokenizer(args)
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
