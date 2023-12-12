import torch
import argparse
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model.visualglm import FineTuneVisualGLMModel
from model.blip2 import BlipImageEvalProcessor
from transformers import AutoTokenizer
from dataset import CovCTRDataset, MimicXrayDataset, FewShotDataset


# class AdjustAdapterMixin(AdapterMixin):
#     @non_conflict
#     def layer_forward(self, hidden_states, mask, old_impl, *args, **kw_args):
#         """
#         hidden_states: [batch, seq_len, hidden_size]
#         mask: [(1, 1), seq_len, seq_len]
#         """
#         layer = self.transformer.layers[kw_args["layer_id"]]
#         # Layer norm at the begining of the transformer layer.
#         hidden_states = layer.input_layernorm(hidden_states)
#         # Self attention.
#         attention_output = layer.attention(hidden_states, mask, **kw_args)

#         attention_output = attention_output + self.ff2[kw_args["layer_id"]](
#             nn.functional.gelu(self.ff1[kw_args["layer_id"]](attention_output))
#         )

#         # Residual connection.
#         layernorm_input = hidden_states + attention_output
#         # Layer norm post the self attention.
#         layernorm_output = layer.post_attention_layernorm(layernorm_input)

#         # MLP.
#         mlp_output = layer.mlp(layernorm_output, **kw_args)
#         mlp_output = mlp_output + self.ff4[kw_args["layer_id"]](
#             nn.functional.gelu(self.ff3[kw_args["layer_id"]](mlp_output))
#         )

#         # Second residual connection.
#         output = layernorm_output + mlp_output

#         return output


# class FineTuneVisualGLMModel(VisualGLMModel):
#     def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
#         super().__init__(
#             args, transformer=transformer, parallel_output=parallel_output, **kw_args
#         )
#         if args.use_ptuning:
#             self.add_mixin(
#                 "ptuning",
#                 PTuningV2Mixin(
#                     args.num_layers,
#                     args.hidden_size // args.num_attention_heads,
#                     args.num_attention_heads,
#                     args.pre_seq_len,
#                 ),
#             )
#         if args.use_lora:
#             self.add_mixin(
#                 "lora",
#                 LoraMixin(
#                     args.num_layers,
#                     args.lora_rank,
#                     layer_range=args.layer_range,
#                 ),
#                 reinit=True,
#             )
#             # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
#         elif args.use_qlora:
#             self.add_mixin(
#                 "lora",
#                 LoraMixin(
#                     args.num_layers,
#                     args.lora_rank,
#                     layer_range=args.layer_range,
#                     qlora=True,
#                 ),
#                 reinit=True,
#             )
#         elif args.use_adapter:
#             # adapter finetune
#             self.add_mixin(
#                 "adapter",
#                 AdjustAdapterMixin(
#                     num_layers=args.num_layers, # 28-transformer一致
#                     hidden_size=args.hidden_size, # 4096
#                     adapter_hidden=args.adapter_hidden, # specified in .sh
#                 ),
#             )
#             pass
#         self.args = args

#     @classmethod
#     def add_model_specific_args(cls, parser):
#         group = parser.add_argument_group(
#             "VisualGLM-finetune", "VisualGLM finetune Configurations"
#         )
#         group.add_argument("--pre_seq_len", type=int, default=8)
#         group.add_argument("--lora_rank", type=int, default=10)
#         group.add_argument("--use_ptuning", action="store_true")
#         group.add_argument("--use_lora", action="store_true")
#         group.add_argument("--use_qlora", action="store_true")
#         group.add_argument("--layer_range", nargs="+", type=int, default=None)
#         group.add_argument("--use_adapter", action="store_true")
#         group.add_argument("--adapter_hidden", type=int, default=128)
#         group.add_argument("--adapter_num_layers", type=int, default=28)
#         group.add_argument("--use_freeze", action="store_true")
#         group.add_argument("--unfreeze_layers", type=str, default="")
#         return super().add_model_specific_args(parser)

#     def disable_untrainable_params(self):
#         enable = []
#         if self.args.use_ptuning:
#             enable.extend(["ptuning"])
#         if self.args.use_lora or self.args.use_qlora:
#             enable.extend(["matrix_A", "matrix_B"])
#         if self.args.use_freeze:
#             unfreeze_layers = self.args.unfreeze_layers.split(',')
#         else:
#             unfreeze_layers = []
#         print('------------unfreeze_layer--------------')
#         for n, p in self.named_parameters():
#             flag = False
#             # adapter unfreeze
#             if self.args.use_adapter and n.startswith("mixins.adapter"):
#                 flag = True
#             elif self.args.use_freeze:
#                 for unfreeze_layer in unfreeze_layers:
#                     if n.startswith(f"transformer.layers.{unfreeze_layer}."):
#                         flag = True
#                         break
#             else:
#                 for e in enable:
#                     if e.lower() in n.lower():
#                         flag = True
#                         break
#             if not flag:
#                 p.requires_grad_(False)
#             else:
#                 print(n)
#         print('------------unfreeze_layer--------------')



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


def create_dataset_function(path, args):
    # tokenizer = get_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
    image_processor = BlipImageEvalProcessor(224)
    dataset = CovCTRDataset(path, image_processor, tokenizer, args)
    # dataset = MimicXrayDataset(path, image_processor, tokenizer, args)
    return dataset


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max_source_length", type=int)
    py_parser.add_argument("--max_target_length", type=int)
    py_parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument("--source_prefix", type=str, default="")
    py_parser.add_argument("--use_trained_eva", action="store_true")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.device = "cpu"


    model_type = "visualglm-6b"
    # if args.ckpt is None:
    # 无存档点 则读入初始预训练参数
    model, args = FineTuneVisualGLMModel.from_pretrained(
        model_type,
        args,
        build_only=True,
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

    if args.use_trained_eva:
        model.mixins['eva'].load_state_dict(
            torch.load(
                f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/eva.ckpt"
            )
        )
    # else:
    #     # 读入微调后的存档点
    #     # args.use_gpu_initialization = True
    #     # args.skip_init=True
    #     # LOCAL_RANK = os.environ.get('LOCAL_RANK')
    #     # args.device=f'cuda:{LOCAL_RANK}'
    #     model, args = FineTuneVisualGLMModel.from_pretrained(
    #         model_type,
    #         args=args,
    #     )
    #     state_dict = torch.load(args.ckpt)
    #     # model = model.to("cuda")
    #     model.load_state_dict(state_dict)


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
